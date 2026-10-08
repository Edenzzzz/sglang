#!/bin/bash
# Analyze cudaDeviceSynchronize distribution in both profiles
export PATH="/mlx_devbox/users/wenxuan.tan/playground/sglang/.venv/bin:$PATH"
cd /mlx_devbox/users/wenxuan.tan/playground/sglang/tasks/debug_kernel_server_perf/profile_logs/nsys

echo "=== Default: cudaDeviceSynchronize distribution ==="
nsys stats --report cuda_api_trace default_mixed_chunk_server.nsys-rep --format csv 2>/dev/null \
  | grep cudaDeviceSynchronize | awk -F',' '{print $2}' | sort -n | awk '
    {a[NR]=$1; s+=$1}
    END {
      n=NR;
      print "Count:", n;
      print "Total (ms):", s/1e6;
      print "Mean (us):", s/n/1e3;
      print "Median (us):", a[int(n/2)]/1e3;
      print "P95 (us):", a[int(n*0.95)]/1e3;
      print "P99 (us):", a[int(n*0.99)]/1e3;
      print "Max (us):", a[n]/1e3;
      # Count by bucket
      lt100=0; lt1000=0; lt10000=0; gt10000=0;
      for(i=1;i<=n;i++){
        if(a[i]<100000) lt100++;
        else if(a[i]<1000000) lt1000++;
        else if(a[i]<10000000) lt10000++;
        else gt10000++;
      }
      print "<100us:", lt100, "100us-1ms:", lt1000, "1ms-10ms:", lt10000, ">10ms:", gt10000;
    }'

echo ""
echo "=== v3: cudaDeviceSynchronize distribution ==="
nsys stats --report cuda_api_trace persistent_v3_server.nsys-rep --format csv 2>/dev/null \
  | grep cudaDeviceSynchronize | awk -F',' '{print $2}' | sort -n | awk '
    {a[NR]=$1; s+=$1}
    END {
      n=NR;
      print "Count:", n;
      print "Total (ms):", s/1e6;
      print "Mean (us):", s/n/1e3;
      print "Median (us):", a[int(n/2)]/1e3;
      print "P95 (us):", a[int(n*0.95)]/1e3;
      print "P99 (us):", a[int(n*0.99)]/1e3;
      print "Max (us):", a[n]/1e3;
      lt100=0; lt1000=0; lt10000=0; gt10000=0;
      for(i=1;i<=n;i++){
        if(a[i]<100000) lt100++;
        else if(a[i]<1000000) lt1000++;
        else if(a[i]<10000000) lt10000++;
        else gt10000++;
      }
      print "<100us:", lt100, "100us-1ms:", lt1000, "1ms-10ms:", lt10000, ">10ms:", gt10000;
    }'
