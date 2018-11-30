#!/usr/bin/env bash


bash ./01_semi_training_update_both_ce.sh
bash ./01_semi_training_update_both_oracle.sh

bash ./01_semi_training_update_lab_ce.sh.sh
bash ./01_semi_training_update_lab_oracle.sh

bash ./01_semi_training_update_unlab_ce.sh
bash ./01_semi_training_update_unlab_oracle.sh