#!/usr/bin/env bash

/usr/bin/streamlink https://cdn-3-go.toya.net.pl:8081/kamery/krak_centrumkongresowe.m3u8 best -o "stream_data/streamlink_$(date +'%Y%m%d_%H%M%S').mp4"

