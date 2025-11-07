"""
    This generates gpu kernel analysis output from nsys rep. Will call nsys
    stats  -r cuda_gpu_kern_trace, get non-overlapped gpu cycles, then generate
    csv and html output for analysis
"""

import argparse
import logging
import os

import regex as re

logger = logging.getLogger(__name__)


# helper data class for annotating kernels
def load_engine_model():
    """returns engine_model built from all json files in the current dir"""
    import glob
    import json

    engine_model = {}

    json_files = glob.glob(os.path.join(os.path.dirname(__file__) or ".", "*.json"))
    for fname in json_files:
        with open(fname, encoding="utf-8") as f:
            engine_model.update(json.load(f))
    return engine_model


class GPUTrace2Graph:
    """
    Parses output of nsys report, generates csv and bar chart output
    """

    def __init__(self):
        import pandas as pd  # avoid importing till needed

        self.pd = pd
        self.pd.options.mode.copy_on_write = True

    # helper functions for generating trace->summary csvs
    def gen_nonoverlapped_sum_from_gputrace(self, in_file, out_file):
        logger.info("loading %s", in_file)
        df = self.pd.read_csv(
            in_file, usecols=["Start (ns)", "Duration (ns)", "Device", "Strm", "Name"]
        )
        df["End (ns)"] = df["Start (ns)"] + df["Duration (ns)"]
        df = self.sum_non_overlapping_intervals(df)
        # get ready to print table with elapsed times per kernel
        df["Instances"] = 1
        df_sum = df.groupby("Name", as_index=False).agg(
            {"Elapsed Time (ns)": "sum", "Duration (ns)": "sum", "Instances": "size"}
        )

        # generate csv
        df_sum["Total Time (sec)"] = df_sum["Duration (ns)"] / 1e9
        df_sum["Elapsed Time (sec)"] = df_sum["Elapsed Time (ns)"] / 1e9
        df_sum = df_sum.sort_values(by="Elapsed Time (sec)", ascending=False)
        df_sum[["Elapsed Time (sec)", "Total Time (sec)", "Instances", "Name"]].to_csv(
            out_file, index=False
        )

    def sum_non_overlapping_intervals(self, df):
        """
        returns new sorted df with Elapsed Time (ns) column using
        vectorized operations
        """
        logger.info("sorting %s trace records by start time", str(df.shape))

        # Sort by start time and reset index
        df = df.sort_values(by="Start (ns)").reset_index(drop=True)

        # Initialize elapsed time as duration
        df["Elapsed Time (ns)"] = df["Duration (ns)"]

        # Get numpy arrays for faster operations
        starts = df["Start (ns)"].values
        ends = df["End (ns)"].values

        # Keep track of current interval end
        current_end = ends[0]
        display_units = max(1, int(len(df) / 100))
        # Update current_end for overlapping intervals
        for i in range(1, len(df)):
            if i % display_units == 0:
                print(f"processing trace: {int(i/len(df) * 100)} %", end="\r")
            if starts[i] <= current_end:
                if ends[i] > current_end:
                    # Partial overlap
                    df.iloc[i, df.columns.get_loc("Elapsed Time (ns)")] = (
                        ends[i] - current_end
                    )
                    current_end = ends[i]
                else:
                    # Complete overlap
                    df.iloc[i, df.columns.get_loc("Elapsed Time (ns)")] = 0
            else:
                # No overlap
                current_end = ends[i]

        return df

    # functions for generating html files
    def make_html(self, df, output_dir, title):
        """make html graph from df"""
        import plotly.express as px

        if df.empty:
            return
        output_name = os.path.join(output_dir, "result")
        if not title:
            title = "Model_Engine"
        x = "Model_Engine"
        y = "Elapsed Time (sec)"
        color = "Category"
        """ generate kernel mapping table  """
        # Sort Model_Engine categories by last field after underscore
        df["Model_Engine"] = self.pd.Categorical(
            df["Model_Engine"],
            sorted(df["Model_Engine"].unique(), key=lambda x: x.split("_")[-1]),
        )
        df[["Model_Engine", color, "Instances", "Name", y]].sort_values(
            by=color
        ).to_csv(f"{output_name}.csv", index=False)
        graph = px.histogram(
            df.round(2),
            x=x,
            y=y,
            title=(f"{y} for {title}"),
            color=color,
            text_auto=True,
        )
        # wrap x axis labels
        graph.update_xaxes(automargin=True)
        graph.write_html(f"{output_name}.html")
        """
            Generate data table with columns per Model_Engine into result.html
        """
        pivot_df = df.pivot_table(
            values="Elapsed Time (sec)",
            index="Category",
            columns="Model_Engine",
            aggfunc="sum",
            observed=False,
        ).round(2)
        # Add sum row at bottom
        pivot_df.loc["total_elapsed_sec"] = pivot_df.sum()
        pivot_df.fillna("").to_html("temp.html")
        with (
            open(f"{output_name}.html", "a", encoding="utf-8") as outfile,
            open("temp.html", encoding="utf-8") as infile,
        ):
            outfile.write(infile.read())
        os.remove("temp.html")

        print(
            f"Finished generating: \n"
            f" {output_name}.html for stack bar chart \n"
            f" {output_name}.csv for Kernel-Category mapping"
        )

    def anno_gpu_kernname(self, df, mapping):
        """add "Category" column"""

        def anno_gpu_kernname_helper(name):
            for kern_name, val in mapping.items():
                if re.search(kern_name, name):
                    return val

        df["Category"] = df["Name"].apply(anno_gpu_kernname_helper)

    def make_nongpu_row(self, df, nongpu_sec):
        """this will append non-gpu time entry at end of df"""
        nongpu_row = self.pd.DataFrame([df.iloc[-1]])
        nongpu_row["Category"] = nongpu_row["Name"] = "CPU(non-GPU)"
        nongpu_row["Instances"] = 1
        nongpu_row["Elapsed Time (sec)"] = nongpu_sec
        return nongpu_row

    def is_valid_file(self, base_file):
        """asserts if base_file is non-existent or is empty"""
        assert (
            os.path.isfile(base_file) and os.path.getsize(base_file) > 0
        ), f"{base_file} doesn't exist or is empty"

    def should_gen_file(self, new_file, base_file):
        """figure out if new file should be generated from base_file"""
        self.is_valid_file(base_file)
        if (
            os.path.exists(new_file)
            and (os.path.getmtime(new_file) > os.path.getmtime(base_file))
            and (os.path.getsize(base_file) > 0)
        ):
            logger.info("reusing %s", new_file)
            return False
        else:
            logger.info("generating %s", new_file)
            return True

    def gen_sum_file(self, file, nsys_cmd):
        """
        generates sum file from nsys trace with times per kernel and
        returns the name of the sum file
        """
        import subprocess

        file_dir = os.path.dirname(file)
        file_name = os.path.basename(file)

        if not file_dir:
            file_dir = "."
        # Walk through trace and get the total non-overlapped time
        nsys_stats_file = os.path.join(file_dir, f"{file_name}_cuda_gpu_trace.csv")
        sum_file = os.path.join(file_dir, f"{file_name}_cuda_gpu_kernel_tracesum.csv")
        if self.should_gen_file(nsys_stats_file, file):
            cmd = [
                nsys_cmd,
                "stats",
                "-r",
                "cuda_gpu_trace",
                file,
                "-o",
                f"{file_dir}/{file_name}",
            ]
            cmd_str = " ".join(cmd)
            logger.info("+ %s", cmd_str)
            # estimate time based on calibrated 240M/min
            file_size_mb = os.path.getsize(file) / 1e6
            logger.info(
                "nsys stats for %.2f MB file expected to take %.2f min",
                file_size_mb,
                file_size_mb / 240,
            )
            try:
                subprocess.run(cmd, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                logger.error(
                    "'%s' failed: %s. Use --nsys_cmd to specify nsys path", cmd_str, e
                )
                exit(1)
            logger.info("generating non-overalapped sum %s", sum_file)
            self.gen_nonoverlapped_sum_from_gputrace(nsys_stats_file, sum_file)
        self.is_valid_file(sum_file)
        logger.info("Finished generating %s", sum_file)
        return sum_file

    def gen_attention_bubble_time(
        self, in_file, out_dir, nsys_cmd, engine_model, bubble_threshold_ns=1000
    ):
        """
        Analyzes CPU/bubble time between GPU kernels in attention operator regions.

        Args:
            in_file: List of (file, engine, model, total_sec) tuples
            out_dir: Output directory
            nsys_cmd: nsys command path
            engine_model: Engine model mapping for kernel categorization
            bubble_threshold_ns: Minimum gap (ns) to consider as bubble time (default: 1000ns = 1us)
        """
        results = []

        for idx, (file, engine, model, total_sec) in enumerate(in_file):
            file_dir = os.path.dirname(file)
            file_name = os.path.basename(file)
            if not file_dir:
                file_dir = "."

            # Generate trace CSV if needed
            file_base = os.path.basename(file).replace(".nsys-rep", "")
            nsys_stats_file = os.path.join(file_dir, f"{file_base}_cuda_gpu_trace.csv")
            if not os.path.exists(nsys_stats_file):
                # Need to generate it first
                self.gen_sum_file(file, nsys_cmd)

            if not os.path.exists(nsys_stats_file):
                logger.warning(f"Trace file not found: {nsys_stats_file}")
                continue

            # Load full trace data
            logger.info(f"Loading trace data from {nsys_stats_file}")
            df = self.pd.read_csv(
                nsys_stats_file,
                usecols=["Start (ns)", "Duration (ns)", "Device", "Strm", "Name"],
            )
            df["End (ns)"] = df["Start (ns)"] + df["Duration (ns)"]

            # Get attention kernel mapping
            if not engine_model.get(engine) or not engine_model[engine].get(model):
                logger.warning(f"Unknown engine/model: {engine}/{model}")
                continue

            mapping = engine_model[engine][model]

            # Categorize kernels
            def get_category(name):
                for kern_name, val in mapping.items():
                    if re.search(kern_name, name):
                        return val
                return "misc"

            df["Category"] = df["Name"].apply(get_category)

            # Filter for attention kernels only
            attn_df = df[df["Category"] == "attn"].copy()

            if len(attn_df) == 0:
                logger.warning(f"No attention kernels found in {file}")
                results.append(
                    {
                        "File": file_name,
                        "Engine": engine,
                        "Model": model,
                        "Total_Attention_Kernels": 0,
                        "Total_Bubble_Time_sec": 0.0,
                        "Total_Attention_Time_sec": 0.0,
                        "Bubble_Percentage": 0.0,
                        "Avg_Bubble_Time_us": 0.0,
                        "Max_Bubble_Time_us": 0.0,
                    }
                )
                continue

            # Sort all attention kernels by start time globally (ignore streams)
            attn_df = attn_df.sort_values(by="Start (ns)").reset_index(drop=True)

            # Calculate bubbles between consecutive attention kernels
            bubble_times = []
            total_attn_time = attn_df["Duration (ns)"].sum()

            if len(attn_df) < 2:
                # Need at least 2 kernels to find gaps
                pass
            else:
                # Find gaps between consecutive kernels globally
                for i in range(len(attn_df) - 1):
                    current_end = attn_df.iloc[i]["End (ns)"]
                    next_start = attn_df.iloc[i + 1]["Start (ns)"]

                    if next_start > current_end:
                        gap = next_start - current_end
                        if gap >= bubble_threshold_ns:
                            bubble_times.append(gap)

            total_bubble_time_ns = sum(bubble_times) if bubble_times else 0
            total_bubble_time_sec = total_bubble_time_ns / 1e9
            total_attn_time_sec = total_attn_time / 1e9

            # Calculate statistics
            bubble_percentage = (
                (total_bubble_time_sec / total_attn_time_sec * 100)
                if total_attn_time_sec > 0
                else 0.0
            )
            avg_bubble_us = (
                (total_bubble_time_ns / len(bubble_times) / 1000)
                if bubble_times
                else 0.0
            )
            max_bubble_us = (max(bubble_times) / 1000) if bubble_times else 0.0

            results.append(
                {
                    "File": file_name,
                    "Engine": engine,
                    "Model": model,
                    "Total_Attention_Kernels": len(attn_df),
                    "Total_Bubble_Time_sec": round(total_bubble_time_sec, 6),
                    "Total_Attention_Time_sec": round(total_attn_time_sec, 4),
                    "Bubble_Percentage": round(bubble_percentage, 2),
                    "Avg_Bubble_Time_us": round(avg_bubble_us, 2),
                    "Max_Bubble_Time_us": round(max_bubble_us, 2),
                    "Num_Bubbles": len(bubble_times),
                }
            )

        if results:
            bubble_df = self.pd.DataFrame(results)
            if out_dir is None:
                out_dir = "."
            else:
                os.makedirs(out_dir, exist_ok=True)
            bubble_file = os.path.join(out_dir, "attention_bubble_time.csv")
            bubble_df.to_csv(bubble_file, index=False)
            logger.info(f"Generated attention bubble time analysis: {bubble_file}")
            print(f"\nAttention Bubble Time Analysis:")
            print(bubble_df.to_string(index=False))
            return bubble_df
        return None

    def gen_kernel_stats(self, in_file, out_dir, nsys_cmd, kernel_name_prefixes):
        """generates kernel statistics for specified kernel name prefixes"""
        results = []
        for idx, (file, engine, model, total_sec) in enumerate(in_file):
            file_dir = os.path.dirname(file)
            file_name = os.path.basename(file)
            if not file_dir:
                file_dir = "."
            sum_file = self.gen_sum_file(file, nsys_cmd)
            # read kernel summary file
            df = self.pd.read_csv(sum_file)

            # Calculate total GPU time (sum of all kernel elapsed times)
            total_gpu_time = df["Elapsed Time (sec)"].sum()

            # Process each kernel name prefix
            for kernel_prefix in kernel_name_prefixes:
                # Filter kernels that start with the prefix
                matching_kernels = df[
                    df["Name"].str.startswith(kernel_prefix, na=False)
                ]

                if len(matching_kernels) == 0:
                    logger.warning(
                        f"No kernels found matching prefix '{kernel_prefix}' in {file}"
                    )
                    total_kernel_time = 0.0
                    percentage = 0.0
                else:
                    # Sum up elapsed time for matching kernels
                    total_kernel_time = matching_kernels["Elapsed Time (sec)"].sum()
                    # Calculate percentage of total GPU time
                    percentage = (
                        (total_kernel_time / total_gpu_time * 100)
                        if total_gpu_time > 0
                        else 0.0
                    )

                results.append(
                    {
                        "File": file_name,
                        "Engine": engine,
                        "Model": model,
                        "Kernel_Prefix": kernel_prefix,
                        "Total_Time_sec": round(total_kernel_time, 4),
                        "Percentage_of_GPU_Time": round(percentage, 2),
                        "Total_GPU_Time_sec": round(total_gpu_time, 4),
                    }
                )

        if results:
            stats_df = self.pd.DataFrame(results)
            if out_dir is None:
                out_dir = "."
            else:
                os.makedirs(out_dir, exist_ok=True)
            stats_file = os.path.join(out_dir, "kernel_stats.csv")
            stats_df.to_csv(stats_file, index=False)
            logger.info(f"Generated kernel statistics: {stats_file}")
            print(f"\nKernel Statistics:")
            print(stats_df.to_string(index=False))
            return stats_df
        return None

    def gen_graph(
        self,
        in_file,
        out_dir,
        title,
        nsys_cmd,
        engine_model,
        kernel_name_prefixes=None,
        analyze_attention_bubbles=False,
        bubble_threshold_ns=1000,
    ):
        """generates graph and csv file from in_file into out_dir"""
        # Generate kernel statistics if kernel_name_prefixes is provided
        if kernel_name_prefixes:
            self.gen_kernel_stats(in_file, out_dir, nsys_cmd, kernel_name_prefixes)

        # Analyze attention bubble time if requested
        if analyze_attention_bubbles:
            self.gen_attention_bubble_time(
                in_file, out_dir, nsys_cmd, engine_model, bubble_threshold_ns
            )

        # Initialize an empty DataFrame to store combined data
        combined_df = self.pd.DataFrame()
        for idx, (file, engine, model, total_sec) in enumerate(in_file):
            file_dir = os.path.dirname(file)
            file_name = os.path.basename(file)
            if not file_dir:
                file_dir = "."
            sum_file = self.gen_sum_file(file, nsys_cmd)
            # read kernel summary file
            df = self.pd.read_csv(sum_file)
            # annotate kernel to their categories
            assert engine_model.get(engine), f"engine {engine} unknown"
            assert engine_model[engine].get(model), f"model {model} unknown"
            # remove nsys-rep from file_name for shorter x-label
            file_name = file_name.replace(".nsys-rep", "")
            df["Model_Engine"] = f"{model}_{engine}_{file_name}_{idx}"
            self.anno_gpu_kernname(df, engine_model[engine][model])
            # patch in non-gpu time
            gpu_sec = round(df["Elapsed Time (sec)"].sum(), 1)
            total_sec = round(float(total_sec), 1)
            if total_sec < gpu_sec:
                logger.warning(
                    "Elapsed sec %.2f < GPU sec %.2f resetting Elapsed sec ",
                    total_sec,
                    gpu_sec,
                )
                total_sec = gpu_sec
            nongpu_row = self.make_nongpu_row(df, total_sec - gpu_sec)
            df = self.pd.concat([df, nongpu_row], ignore_index=True)
            combined_df = self.pd.concat([combined_df, df], ignore_index=True)
        if out_dir is None:
            out_dir = "."
        else:
            os.makedirs(out_dir, exist_ok=True)
        # generate html file
        self.make_html(combined_df, out_dir, title)


def parse_tuple(s):
    return tuple(s.split(","))


def main():
    logging.basicConfig(
        format=("%(asctime)s - %(levelname)s - %(message)s"), level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description=(
            "Process nsys rep and generate kernel non-overlapped cycles. \n"
            "Example:\n"
            "gputrc2graph.py --in_file d1.nsys-rep,sglang,llama,100 \n"
            "d2.nsys-rep,sglang,gpt-oss,102 "
            '--out_dir results/ --title "Model=gpt-oss SGLANG chart"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # load supported engine_model
    engine_model_supported = load_engine_model()
    # Get a string representation of supported engine/model combinations
    engine_model_supported_str = ", ".join(
        f"{engine}:[{', '.join(models.keys())}]"
        for engine, models in engine_model_supported.items()
    )
    parser.add_argument(
        "--in_file",
        type=parse_tuple,
        nargs="+",
        help=(
            "list of (nsys-rep, engine, model, elapsed_nonprofiled_sec) "
            "separated by space. Elapsed_nonprofiled_sec is runtime without "
            "profiling used to calculate non-gpu time. Specify 0 to use "
            "elapsed time from nsys-rep but that might inflate non-gpu time. "
            f"Available engine:[model] are: {engine_model_supported_str} "
            f"Example: --infile d1.nsys-rep,sglan,llama,100 "
            "d2.nsys-rep,sglang,gpt-oss,102"
        ),
        required=True,
    )
    parser.add_argument("--out_dir", help=("output dir for result.csv/html"))
    parser.add_argument("--title", help=("title for html chart"))
    parser.add_argument(
        "--nsys_cmd",
        help=("nsys cmd, e.g. /usr/bin/nsys, Default: nsys"),
        default="nsys",
    )
    parser.add_argument(
        "--kernel_name_prefixes",
        type=str,
        nargs="*",
        default=["void flashinfer::PersistentKernelTemplate"],
        help=(
            "List of kernel name prefixes to analyze. "
            "Kernels whose names start with any of these prefixes will be included. "
            "Default: 'void flashinfer::PersistentKernelTemplate'"
        ),
    )
    parser.add_argument(
        "--analyze_attention_bubbles",
        action="store_true",
        help=(
            "Analyze CPU/bubble time between GPU kernels in attention operator regions. "
            "Outputs attention_bubble_time.csv with bubble time statistics."
        ),
    )
    parser.add_argument(
        "--bubble_threshold_ns",
        type=int,
        default=1000,
        help=(
            "Minimum gap in nanoseconds to consider as bubble time between kernels. "
            "Default: 1000 (1 microsecond)"
        ),
    )
    args = parser.parse_args()
    gputrace = GPUTrace2Graph()
    gputrace.gen_graph(
        args.in_file,
        args.out_dir,
        args.title,
        args.nsys_cmd,
        engine_model_supported,
        args.kernel_name_prefixes,
        args.analyze_attention_bubbles,
        args.bubble_threshold_ns,
    )


if __name__ == "__main__":
    main()
