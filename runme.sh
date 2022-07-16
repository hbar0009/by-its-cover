module load python/3.7.3-system

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Finished installing, downloading data"

python3 download_images.py "data/train" "book30-listing-train.csv"
python3 download_images.py "data/test" "book30-listing-test.csv"

echo "Finished downloading data, now training model"

python3 bookcover_genre_predict.py

echo "Script complete"
