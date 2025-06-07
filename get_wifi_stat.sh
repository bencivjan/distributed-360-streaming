#!/bin/bash

while true; do
    # Command to get the station dump
    output=$(iw dev wlan0 station dump)

    # Extract the signal line
    signal_line=$(echo "$output" | grep "signal:")

    # Extract the bitrate lines
    tx_bitrate_lines=$(echo "$output" | grep "tx bitrate:")

    rx_bitrate_lines=$(echo "$output" | grep "rx bitrate:")

    mac_address_line=$(echo "$output" | grep "Station")

    # Print the signal line with timestamp to the terminal
    echo "$(date): $signal_line"
    echo "$(date): $tx_bitrate_lines"
    echo "$(date): $rx_bitrate_lines"
    echo "$(date): $mac_address_line"

    # Log the signal line and bitrate lines with timestamp to wifi.log
    echo "$(date): $signal_line" >> wifi.log 
    echo "$(date): $tx_bitrate_lines" >> wifi.log
    echo "$(date): $rx_bitrate_lines" >> wifi.log
    echo "$(date): $mac_address_line" >> wifi.log

    # Sleep for 1 second
    sleep 1
done

