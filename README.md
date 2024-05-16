# LifeHD
[IPSN 2024] Lifelong Intelligence Beyond the Edge using Hyperdimensional Computing

[[arXiv link]](https://arxiv.org/html/2403.04759v1)

## File Structure


```
.
├── LICENSE
├── README.md           // this file
├── main.py             // main file
├── methods             // implementation of LifeHD
├── requirements.txt
├── scripts             // scripts to run each method
└── utils               // utils files
```

## Prerequisites

We test with Python3.8. We recommend using conda environments:

```bash
conda create --name lifehd-py38 python=3.8
conda activate lifehd-py38
python3 -m pip install -r requirements.txt
```

All require Python packages are included in `requirements.txt` and can be installed automatically.


### Dataset Preparation

As mentioned in the paper, we experiment on MHEALTH, ESC-50 and CIFAR-100.
This repo also includes the dataset setup for MNIST, CIFAR-10, HAR (HAR-timeseries) and ISOLET.

* We preprocess the [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) and [HAR-timeseries](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) data from their original data. HAR-timeseries is the raw time series version of the dataset. To use, please download the corresponding files and place them in the appropriate path.

  * To use MHEALTH
    ```bash
    mkdir datasets # create the datasets folder if it does not exist
    mkdir datasets/MHEALTH # create the folder for MHEALTH
    ```

    Download the preprocessed MHEALTH data, [mhealth.mat](https://drive.google.com/file/d/1IcSuRKvw82n8e_t4Iq6u2i1ro_9GoGqL/view?usp=sharing), and place it inside the MHEALTH folder.

  * To use HAR-timeseries

    ```bash
    mkdir datasets # create the datasets folder if it does not exist
    mkdir datasets/HAR_TimeSeries # create the folder for HAR-timeseries
    ```

    Download the preprocessed HAR time series data, [har.mat](https://drive.google.com/file/d/12B_cKdRSv-pgimTqBgu0V4ac4Y0EzQHP/view?usp=sharing), and place it inside the HAR_TimeSeries folder.

* We adapt the [ESC-50](https://github.com/karolpiczak/ESC-50) data. To simplify the setup, we offer our processed version. Please download [esc50.zip](https://drive.google.com/file/d/1oFNIzO6JR3JfcpGN4gQS3qjxQrRGnadM/view?usp=sharing) and place it under the following newly created folder

  ```bash
  mkdir datasets/esc50 # create the folder for ESC-50
  ```

The rest datasets should be downloaded and configured automatically by our group.

### Model Preparation

In the HDnn encoding, we need a pretrained and frozen neural network as feature extractor. For CIFAR-10/CIFAR-100, we offer pretrained ResNets and MobileNets downloaded from torchvision. For ESC-50, we employ the pretrained ACDNet adapt from their [Github repo](https://github.com/mohaimenz/acdnet/tree/master).

To setup these pretrained model, please download [pretrained_models.zip](https://drive.google.com/file/d/1DMd_YvEPweZsM3bkcmmckA26dNeJbCGn/view?usp=sharing) and unzip it under `utils/`.


## Getting Started


We provide our scripts for running various experiments in the paper in `scripts`:

* To run the supervised HDC

  * MNIST/CIFAR-10/CIFAR-100/ESC-50/HAR

  ```bash
  bash run_basichd.sh BasicHD <cifar10/cifar100/mhealth/esc-50/har> <iid/seq> <trial ID> idlevel
  ```

  `<iid/seq>` specifies the data stream order. `idlevel` configures the encoding method, which indicates a simpler version of the spatiotemporal encoding.

  For example, trial 0 of the CIFAR-10, class-incremental streams experiment could be fired with
  ```bash
  bash run_basichd.sh BasicHD cifar10 seq 0 idlevel
  ```

  * MHEALTH/HAR_timeseries

  ```bash
  bash run_basichd.sh BasicHD <mhealth/har_timeseries> <iid/seq> <trial ID> spatiotemporal
  ```

  The difference is the last argument - we set to use the full `spatiotemporal` encoding here.

* To run LifeHD

  * MNIST/CIFAR-10/CIFAR-100/ESC-50

  ```bash
  bash run_lifehd.sh LifeHD <cifar10/cifar100/mhealth/esc-50> <iid/seq> <trial ID> idlevel
  ```

  * MHEALTH/HAR_timeseries

  ```bash
  bash run_lifehd_timeseries.sh LifeHD <mhealth/har_timeseries> <iid/seq> <trial ID> spatiotemporal
  ```

* To run SemiHD

  * MNIST/CIFAR-10/CIFAR-100/ESC-50

  ```bash
  bash run_semihd.sh SemiHD <cifar10/cifar100/mhealth/esc-50> <iid/seq> <trial ID> idlevel <label_ratio>
  ```

  * MHEALTH/HAR_timeseries

  ```bash
  bash run_semihd.sh SemiHD <mhealth/har_timeseries> <iid/seq> <trial ID> spatiotemporal <label_ratio>
  ```

  The commands are very similar to the previous cases, except adding an additional argument for labeling ratio in the end.

* To run LifeHDsemi

  * MNIST/CIFAR-10/CIFAR-100/ESC-50

  ```bash
  bash run_lifehdsemi.sh LifeHDsemi <cifar10/cifar100/mhealth/esc-50> <iid/seq> <trial ID> idlevel <label_ratio>
  ```

  * MHEALTH/HAR_timeseries

  ```bash
  bash run_lifehdsemi.sh LifeHDsemi <mhealth/har_timeseries> <iid/seq> <trial ID> spatiotemporal <label_ratio>
  ```

* To run LifeHDa
  LifeHDa can be fired with the same scripts mentioned above by appropriately configuring the folloiwng arguments in each script:

  * `--mask_mode`: fixed or adaptive
  * `--mask_dim`: dimension of the mask

## License

MIT

If you have any questions, please feel free to contact x1yu@ucsd.edu.