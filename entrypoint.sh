#!/bin/sh
# Override user ID lookup to cope with being randomly assigned IDs using
# the -u option to 'docker run'.

USER_ID=$(id -u)
GROUP_ID=$(id -g)

if [ x"$USER_ID" != x"0" ]; then
    NSS_WRAPPER_PASSWD=/tmp/passwd.nss_wrapper
    NSS_WRAPPER_GROUP=/tmp/group.nss_wrapper
    cat /etc/passwd > $NSS_WRAPPER_PASSWD
    cat /etc/group > $NSS_WRAPPER_GROUP
    echo "fun3d:x:$USER_ID:0:FUN3D,,,:/work:/bin/bash" >> $NSS_WRAPPER_PASSWD
    echo "fun3d:x:$GROUP_ID:" >> $NSS_WRAPPER_GROUP
    export NSS_WRAPPER_PASSWD
    export NSS_WRAPPER_GROUP
    LD_PRELOAD=/usr/lib/libnss_wrapper.so
    export LD_PRELOAD
fi
export PATH=/work/kaldi/src/online2bin:$PATH
exec $*

