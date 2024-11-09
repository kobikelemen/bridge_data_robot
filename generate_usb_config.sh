#!/bin/bash

# Initialize the YAML variables to empty strings
blue=""
yellow=""
wrist=""
d435=""

# Run v4l2-ctl to fetch devices and parse them line by line
while IFS= read -r line; do
    # Check for the device identifiers and store them accordingly
    if [[ $line == *"Piwebcam: UVC Camera"* ]]; then
        wrist=$(echo "$line" | awk -F '(' '{print $2}' | awk -F ')' '{print $1}')
    elif [[ $line == *"HD Pro Webcam C920"* ]] && [ -z "$blue" ]; then
        blue=$(echo "$line" | awk -F '(' '{print $2}' | awk -F ')' '{print $1}')
    elif [[ $line == *"HD Pro Webcam C920"* ]]; then
        yellow=$(echo "$line" | awk -F '(' '{print $2}' | awk -F ')' '{print $1}')
    elif [[ $line == *"Intel(R) RealSense(TM) Depth Ca"* ]]; then
        d435=$(echo "$line" | awk -F 'Ca ' '{print $2}' | awk -F ')' '{print $1}')
    elif [[ $line == *"Arducam B0459"* ]]; then
        # If blue is not set, use Arducam as blue camera
        if [ -z "$blue" ]; then
            # blue=$(echo "$line" | awk -F '(' '{print $2}' | awk -F ')' '{print $1}')
            # Get the next line which should contain the device path
            read -r device_path
            # blue=$(echo "$device_path" | tr -d '\t')
            # Only use /dev/video0 for this camera
            if [[ $device_path == *"/dev/video0"* ]]; then
                blue="/dev/video0"
            fi
        fi
    fi
done < <(v4l2-ctl --list-devices)

# Print the generated YAML format
cat << EOF > usb_connector_chart.yml
blue: '$blue'
yellow: '$yellow'
wrist: '$wrist'
D435: '$d435'
EOF
