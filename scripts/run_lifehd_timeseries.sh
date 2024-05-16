# Usage:
#   Run HD on various dataset under five streaming settings: iid, seq, seq-bl, seq-cc, seq-im
#   Example usage:
#     ./run_lifehd_timeseries.sh LifeHD mnist iid trial# spatiotemporal
#   Method choices: BasicHD, SemiHD, LifeHD, LifeHDsemi
#   Dataset choices: mnist, cifar10, cifar100, har, har_timeseries, mhealth, esc50
#   Trial #: the number of trial
#   Encoder choices: none, rp, idlevel, spatiotemporal

cd ..;

if [ "$2" = "har_timeseries" ]; then
  batch_size=32
  val_batch_size=32
  feat_ext=none
  hd_dim=1000
  #encoder="spatiotemporal"
  levels=10
  randomness=0.2
  flipping=0.01
  k_merge_min=6
  merge_freq=25
elif [ "$2" = "mhealth" ]; then
  batch_size=32
  val_batch_size=32
  feat_ext=none
  hd_dim=1000
  #encoder="spatiotemporal"
  levels=10
  randomness=0.2
  flipping=0.01
  k_merge_min=20
  merge_freq=25
fi

mask_dim=10000
mask_mode="fixed"
merge_mode="merge"
beta=3.0
mem=50

if [ "$2" = "mhealth" ]; then
  if [ "$3" = "iid" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type iid  --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --mask_dim $mask_dim --mask_mode $mask_mode --merge_mode $merge_mode \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --mask_dim $mask_dim --mask_mode $mask_mode --merge_mode $merge_mode \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-bl" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-cc" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --mask_dim $mask_dim --mask_mode $mask_mode --merge_mode $merge_mode \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-im" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --mask_dim $mask_dim --mask_mode $mask_mode --merge_mode $merge_mode \
      --win_size 128 --overlap 0.75
  fi
fi


if [ "$2" = "har_timeseries" ]; then
  if [ "$3" = "iid" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type iid  --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --mask_dim $mask_dim --mask_mode $mask_mode --merge_mode $merge_mode \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --mask_dim $mask_dim --mask_mode $mask_mode --merge_mode $merge_mode \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-bl" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --mask_dim $mask_dim --mask_mode $mask_mode --merge_mode $merge_mode \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-cc" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --mask_dim $mask_dim --mask_mode $mask_mode --merge_mode $merge_mode \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-im" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --max_classes $mem --beta $beta --k_merge_min $k_merge_min --merge_freq $merge_freq \
      --mask_dim $mask_dim --mask_mode $mask_mode --merge_mode $merge_mode \
      --win_size 128 --overlap 0.75
  fi
fi
