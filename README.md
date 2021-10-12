# Semi-supervised-img-classification
Semi-supervised image classification methods to leverage unlabelled data
This repository focuses on experimenting with different semi-supervised methods for image classifications. The methods included are:
 * Pi Model ([Paper](https://arxiv.org/abs/1610.02242))
 * Temporal Ensembling ([Paper](https://arxiv.org/abs/1610.02242))
 * Exponential Moving Average ([Paper](https://arxiv.org/abs/1703.01780))

## Image scraping for unlabelled data
To scrape image data for unlabelled data portion. Use the `img_scrapper.py`. This script will access google image and download images
related to a provided query.

```bash
python3 img_scrapper --query <query> 
					--driver <driver>
					--dir <dir>
					--num-imgs <num_imgs>
```

Where : 
 * query : The query provided to Google image.
 * driver : Path to the web driver (Chrome)
 * dir : The directory to save the downloaded images
 * num-imgs : Number of images to download
