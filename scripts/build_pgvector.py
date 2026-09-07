import subprocess
import sys
import os
import shutil

PGROOT = r"C:\Program Files\PostgreSQL\18"
TEMP_DIR = os.path.join(os.environ.get("TEMP", r"C:\Temp"), "pgvector_build")
PGVECTOR_TAG = "v0.8.0"


def run(cmd, cwd=None, env=None):
    print(f">> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


def find_cl():
    import glob
    patterns = [
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return os.path.dirname(matches[0])
    return None


def main():
    cl_dir = find_cl()
    if not cl_dir:
        print("ERROR: MSVC compiler (cl.exe) not found.")
        print("Install Visual Studio 2022 Build Tools with C++ workload first.")
        sys.exit(1)

    print(f"Found MSVC: {cl_dir}")

    vc_root = os.path.normpath(os.path.join(cl_dir, "..", "..", "..", ".."))
    vcvars = os.path.join(vc_root, "Auxiliary", "Build", "vcvars64.bat")

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    rc = run(f'git clone --branch {PGVECTOR_TAG} --depth 1 https://github.com/pgvector/pgvector.git "{TEMP_DIR}"')
    if rc != 0:
        print("Git clone failed.")
        sys.exit(1)

    build_cmd = (
        f'call "{vcvars}" && '
        f'set PGROOT={PGROOT} && '
        f'nmake /F Makefile.win && '
        f'nmake /F Makefile.win install'
    )

    rc = run(build_cmd, cwd=TEMP_DIR)
    if rc != 0:
        print("Build failed.")
        sys.exit(1)

    print("\npgvector built and installed successfully!")
    print("Now run in psql: CREATE EXTENSION IF NOT EXISTS vector;")


if __name__ == "__main__":
    main()
