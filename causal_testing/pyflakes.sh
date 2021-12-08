# run `bash pyflakes.sh` to run pyflakes locally before committing
find . -type f -name "*.py" | xargs -n 1 pyflakes
