# Set the number of times to run the script
num_times=50

for ((i=1; i<=$num_times; i++)); do
	echo "===== Iteration $i ====="
	python analyze_runtimes_compilation.py
done
