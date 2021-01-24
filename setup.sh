#!/bin/bash
sudo ./traffic-control.sh -r
sudo ./traffic-control.sh --uspeed=$1 --dspeed=$2 -d=20 172.31.0.0/16
sudo tc filter show dev ens3
sudo tc qdisc show dev ens3
