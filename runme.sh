module load python/3.7.3-system

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# this snippet here was provided by the book cover dataset
# available at: https://github.com/uchidalab/book-dataset/blob/master/scripts/download_images.sh
OUTPUT_DIRPATH="images"
CSV_FILEPATH="Task2/book32-listing.csv"

python3 download_images.py ${OUTPUT_DIRPATH} ${CSV_FILEPATH}

python3 bookcover_genre_predict.py
