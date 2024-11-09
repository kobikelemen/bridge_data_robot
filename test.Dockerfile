FROM ubuntu:20.04

# First create any missing groups
RUN groupadd -f input

# Create robonet user and add to groups
RUN useradd -m robonet && \
    usermod -aG dialout,video,input,tty robonet

# Let's also check what groups exist
RUN echo "=== Available Groups ===" && cat /etc/group

# Same diagnostic script
RUN echo '#!/bin/bash\n\
echo "=== User and Group Info ==="\n\
id\n\
echo -e "\\n=== Groups ==="\n\
groups\n\
echo -e "\\n=== Device Permissions ==="\n\
ls -l /dev/tty* /dev/video* 2>/dev/null\n\
echo -e "\\n=== Console Access ==="\n\
ls -l /dev/console\n\
echo -e "\\n=== Current Directory Permissions ==="\n\
ls -la\n\
echo -e "\\n=== Docker Socket ==="\n\
ls -l /var/run/docker.sock 2>/dev/null\n\
echo -e "\\n=== Process List ==="\n\
ps aux | grep -v "\[" | head -n 5' > /check-permissions && chmod +x /check-permissions

USER robonet
CMD ["/check-permissions"]