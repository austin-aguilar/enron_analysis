# enron_analysis

# Quick Start

1. Clone the repository
2. Install lfs: `brew install git-lfs`
3. Install lfs into repo: `git lfs install`
4. Pull lfs data: `git lfs pull`
5. Decompress the data: `unzip data.zip`
6. Create a virtual enviornment
7. Install pandas: `pip install pandas`
8. Run `python data/process_emails.ipynb`
9. Create a python notebook and import `data/processed_emails.csv`
    - The data has columns: text, sender, recipient1, recipient2, recipient3, Subject, folder, and date
10. Analyze the data as needed