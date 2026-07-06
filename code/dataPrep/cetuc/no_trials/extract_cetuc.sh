#!/bin/bash
#SBATCH --account=def-aravila
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=extract_cetuc
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

# Root directory of the downloaded CETUC dataset
ROOT="/home/amoroso/links/scratch/CORPORA_DIR/CETUC/data"

echo "Searching for .tar.gz archives under $ROOT..."

find "$ROOT" -type f -name "*.tar.gz" | while read -r archive; do
    dir=$(dirname "$archive")

    echo "Extracting: $archive"

    if tar -xzf "$archive" -C "$dir"; then

        # Remove AppleDouble metadata files (macOS resource forks)
        find "$dir" -type f -name "._*" -delete

        # Remove transcription files
        find "$dir" -type f -name "*.txt" -delete

        # Remove the archive after successful extraction
        rm "$archive"

        echo "✓ Extracted and cleaned: $archive"

    else
        echo "✗ Failed: $archive"
    fi
done

echo "Extraction complete."

# #!/bin/bash
# #SBATCH --account=def-aravila
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=32G
# #SBATCH --time=0-01:00:00
# #SBATCH --job-name=extract_cetuc
# #SBATCH --error=%x.%j.err
# #SBATCH --output=%x.%j.out

# # Root directory of the downloaded CETUC dataset
# ROOT="/home/amoroso/links/scratch/CORPORA_DIR/CETUC/data"

# echo "Searching for .tar.gz archives under $ROOT..."

# find "$ROOT" -type f -name "*.tar.gz" | while read -r archive; do
#     dir=$(dirname "$archive")

#     echo "Extracting: $archive"

#     tar -xzf "$archive" -C "$dir"

#     if [ $? -eq 0 ]; then
#         echo "✓ Done"
#         rm "$archive"
#     else
#         echo "✗ Failed: $archive"
#     fi
# done

# echo "Extraction complete."