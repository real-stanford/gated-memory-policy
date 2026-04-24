#!/bin/bash

ports=(
    16666
    16667
    16700
    16701
    15557
    15558
    18765
    5555
    5556
)


for port in "${ports[@]}"; do
    port_pid=$(lsof -i :$port | tail -n +2 | awk '{print $2}')
    if [ -n "$port_pid" ]; then
        echo "Killing process on port $port, pid $port_pid"
        kill -SIGINT $port_pid
    fi
done
