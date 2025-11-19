# %%
import pandas as pd
import os
# %%
folder_path = r"/Users/Lenovo/Desktop/BioHack_Mun/data"
# %%
HEK_10uM_counts_path = os.path.join(folder_path, "MCE_Bioactive_Compounds_HEK293T_10μM_Counts.xlsx")
HEK_10uM_meta_path = os.path.join(folder_path, "MCE_Bioactive_Compounds_HEK293T_10μM_MetaData.xlsx")
TCM_HEK_20_meta_path = os.path.join(folder_path, "TCM_Compounds_HEK293T_20_MetaData.xlsx")
# %%
HEK_10uM_counts_df = pd.read_excel(HEK_10uM_counts_path, nrows = 100) # sheet_name
HEK_10uM_meta_df = pd.read_excel(HEK_10uM_meta_path, nrows = 100) # sheet_name
HEK_10uM_meta_df.columns = HEK_10uM_meta_df.iloc[0]  # set first row as column names
HEK_10uM_meta_df = HEK_10uM_meta_df[1:].reset_index(drop=True)
# %%
''' ------------------------- blank number count in treatment columns per dataset -----------------'''
blank_count = HEK_10uM_meta_df["treatment"].str.startswith("Blank").sum()
print(f"Number of values starting with 'Blank': {blank_count}")
# %%
for filename in os.listdir(folder_path):
    if "Meta" in filename and filename.endswith(".xlsx"):
        file_path = os.path.join(folder_path, filename)
        
        df = pd.read_excel(file_path)

        # Fix headers: first row as column names
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        
        # Determine which column exists
        if "treatment" in df.columns:
            col_name = "treatment"
        elif "Treat" in df.columns:
            col_name = "Treat"
        else:
            print(f"{filename}: Neither 'treatment' nor 'Treat' column found")
            continue  # skip to next file
        
        # Count values starting with "Blank" in 'treatment' column
        blank_count = df[col_name].str.startswith("Blank").sum()
        row_count = len(df)
        print(f"{filename}: Number of values starting with 'Blank' = {blank_count}")
        print(f"{filename}: Number of compounds = {row_count}")

# %%
TCM_HEK_20_meta_df = pd.read_excel(TCM_HEK_20_meta_path, nrows = 100) # sheet_name
TCM_HEK_20_meta_df.columns = TCM_HEK_20_meta_df.iloc[0]  # set first row as column names
TCM_HEK_20_meta_df = TCM_HEK_20_meta_df[1:].reset_index(drop=True)
# %%
