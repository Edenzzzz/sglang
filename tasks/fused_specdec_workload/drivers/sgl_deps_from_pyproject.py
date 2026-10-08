import tomllib
d = tomllib.load(open('/dev/shm/w/python/pyproject.toml', 'rb'))
deps = d['project'].get('dependencies', [])
skip = ('torch', 'flashinfer', 'sgl-kernel', 'sgl_kernel', 'triton', 'torchao', 'torchaudio',
        'torchvision', 'xgrammar', 'cuda-python', 'nvidia-')
keep = [x for x in deps if not x.replace('_', '-').lower().startswith(skip)]
with open('/dev/shm/w/sgl_deps.txt', 'w') as f:
    f.write("\n".join(keep) + "\n")
print(len(deps), 'declared;', len(keep), 'to install (kernel stack excluded)')
