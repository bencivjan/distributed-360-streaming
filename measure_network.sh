#!/bin/bash

# Check if a file name argument is provided
if [ $# -le 1 ]; then
    echo "Usage: $0 <c/s (client or server)> <output_file> <server IP (if client)>"
    exit 1
fi

# Assign the first argument to the output_file variable
role=$1
output_file=$2
server_ip=$3

if [[ "$role" == "c" ]]; then

    # Run the iperf3 server and write the output to the specified file
    echo "Starting iperf3 client..."
    echo "Output is being written to $output_file"

    iperf3 -c $server_ip -i 1 -V -t 30 2>&1 | tee -a $output_file
    iperf3 -c $server_ip -i 1 -V -u -b 10M -t 30 2>&1 | tee -a $output_file

elif [[ "$role" == "s" ]]; then

    # Run the iperf3 server and write the output to the specified file
    echo "Starting iperf3 server..."
    iperf3 -s -V 2>&1 | tee $output_file

    echo "Output is being written to $output_file"

else
    echo "Input either 'c' or 's' as first argument"
    exit 1
fi