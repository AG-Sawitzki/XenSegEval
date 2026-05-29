#!/bin/bash
#
. ~/.bashrc
#
#install cargo using rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#
#install proseg using cargo
cargo install proseg
