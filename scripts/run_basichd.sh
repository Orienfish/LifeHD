# Usage:
#   Run HD on various dataset under five streaming settings: iid, seq, seq-bl, seq-cc, seq-im
#   Example usage:
#     ./run_basichd.sh BasicHD mnist iid trial# idlevel
#   Method choices: BasicHD, SemiHD, LifeHD, LifeHDsemi
#   Dataset choices: mnist, cifar10, cifar100, har, har_timeseries, mhealth, esc50
#   Trial #: the number of trial
#   Encoder choices: none, rp, idlevel, spatiotemporal

cd ..;

if [ "$2" = "mnist" ] || [ "$2" = "har" ] || 
  [ "$2" = "isolet" ]; then
  batch_size=64
  val_batch_size=64
  feat_ext=none
  hd_dim=10000
  #encoder="idlevel"
  levels=100
  randomness=0.2
  flipping=0.01
elif [ "$2" = "cifar10" ] || [ "$2" = "cifar100" ] || 
  [ "$2" = "tinyimagenet" ] || [ "$2" = "core50" ] || 
  [ "$2" = "stream51" ]; then
  batch_size=32
  val_batch_size=32
  feat_ext=mobilenet_v2  # resnet18
  pretrained_on=imagenet
  hd_dim=10000
  #encoder="idlevel"
  levels=100
  randomness=0.2
  flipping=0.01
elif [ "$2" = "har_timeseries" ]; then
  batch_size=32
  val_batch_size=32
  feat_ext=none
  hd_dim=1000
  #encoder="timeseries"
  levels=10
  randomness=0.2
  flipping=0.01
elif [ "$2" = "mhealth" ]; then
  batch_size=32
  val_batch_size=32
  feat_ext=none
  hd_dim=1000
  #encoder="timeseries"
  levels=10
  randomness=0.2
  flipping=0.01
elif [ "$2" = "esc50" ]; then
  batch_size=32
  val_batch_size=32
  feat_ext=acdnet
  hd_dim=10000
  feat_ext_ckpt="utils/pretrained_models/acdnet_20khz_trained_model_fold4_91.00.pt"
  #encoder="idlevel"
  levels=100
  randomness=0.2
  flipping=0.02
fi

if [ "$2" = "mnist" ]; then
  if [ "$3" = "iid" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type iid --batch_size "$batch_size" --val_batch_size "$val_batch_size" --num_workers 1 \
      --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi

  if [ "$3" = "seq" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --batch_size "$batch_size" --val_batch_size "$val_batch_size" --num_workers 1 \
      --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi

  if [ "$3" = "seq-bl" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi

  if [ "$3" = "seq-cc" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi

  if [ "$3" = "seq-im" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
    --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi
fi


if [ "$2" = "har" ]; then
    if [ "$3" = "iid" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type iid  --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
    fi

    if [ "$3" = "seq" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type class_iid --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
    fi

    if [ "$3" = "seq-bl" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type class_iid --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
    fi

    if [ "$3" = "seq-cc" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type class_iid --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
    fi

    if [ "$3" = "seq-im" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type class_iid --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
    fi
fi


if [ "$2" = "isolet" ]; then
  if [ "$3" = "iid" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type iid  --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi

  if [ "$3" = "seq" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi

  if [ "$3" = "seq-bl" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi

  if [ "$3" = "seq-cc" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi

  if [ "$3" = "seq-im" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4"
  fi
fi


if [ "$2" = "mhealth" ]; then
  if [ "$3" = "iid" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type iid  --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_instance --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-bl" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_instance --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-cc" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_instance --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-im" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_instance --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi
fi



if [ "$2" = "har_timeseries" ]; then
  if [ "$3" = "iid" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type iid  --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_instance --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-bl" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_instance --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-cc" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_instance --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi

  if [ "$3" = "seq-im" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_instance --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 0.8 --test_samples_ratio 0.2 --trial "$4" \
      --win_size 128 --overlap 0.75
  fi
fi


if [ "$2" = "cifar10" ] ; then
    if [ "$3" = "iid" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type iid  --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
        --pretrained_on "$pretrained_on"
    fi

    if [ "$3" = "seq" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type class_iid --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
        --pretrained_on "$pretrained_on"
    fi

    if [ "$3" = "seq-bl" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type class_iid --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
        --pretrained_on "$pretrained_on"
    fi

    if [ "$3" = "seq-cc" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type class_iid --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
        --pretrained_on "$pretrained_on"
    fi

    if [ "$3" = "seq-im" ]; then
      python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
        --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
        --training_data_type class_iid --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
        --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
        --pretrained_on "$pretrained_on"
    fi
fi


if [ "$2" = "cifar100" ] ; then
  if [ "$3" = "iid" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type iid  --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 8 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --pretrained_on "$pretrained_on"
  fi

  if [ "$3" = "seq" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 8 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --pretrained_on "$pretrained_on"
  fi

  if [ "$3" = "seq-bl" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 8 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --pretrained_on "$pretrained_on"
  fi

  if [ "$3" = "seq-cc" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 8 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --pretrained_on "$pretrained_on"
  fi

  if [ "$3" = "seq-im" ]; then
    python main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 8 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --pretrained_on "$pretrained_on"
  fi
fi

if [ "$2" = "esc50" ] ; then
  if [ "$3" = "iid" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type iid  --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --feature_ext_ckpt "$feat_ext_ckpt"
  fi

  if [ "$3" = "seq" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --feature_ext_ckpt "$feat_ext_ckpt"
  fi

  if [ "$3" = "seq-bl" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --blend_ratio 0.5 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --feature_ext_ckpt "$feat_ext_ckpt"
  fi

  if [ "$3" = "seq-cc" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --n_concurrent_classes 2 --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --feature_ext_ckpt "$feat_ext_ckpt"
  fi

  if [ "$3" = "seq-im" ]; then
    python3 main.py --method "$1" --dataset "$2" --feature_ext "$feat_ext" --hd_encoder "$5" \
      --dim "$hd_dim" --num_levels "$levels" --randomness "$randomness" --flipping "$flipping" \
      --training_data_type class_iid --imbalanced --batch_size "$batch_size" --val_batch_size "$val_batch_size" \
      --num_workers 1 --epochs 1 --train_samples_ratio 1.0 --test_samples_ratio 1.0 --trial "$4" \
      --feature_ext_ckpt "$feat_ext_ckpt"
  fi
fi