import pandas as pd
import os
import re
import pdfplumber

data = []  # store final rows

# month name → number
months = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# loop all files
for f in sorted([x for x in os.listdir(".") if x.endswith((".xlsx", ".pdf"))]):

    if f.endswith(".xlsx"):
        df = pd.read_excel(f)  # read excel

        txt = str(df.columns[0])  # extract header text
        yr = re.search(r"20\d{2}", txt)
        year = int(yr.group()) if yr else 2024
        month = next((months[m] for m in months if m in txt.lower()), 1)

        headers = df.iloc[0].values  # fix headers
        df.columns = headers
        df = df.iloc[1:]

    else:
        # read table from pdf
        with pdfplumber.open(f) as pdf:
            table = pdf.pages[0].extract_table()
        df = pd.DataFrame(table[1:], columns=table[0])

        txt = f  # use filename
        yr = re.search(r"20\d{2}", txt)
        year = int(yr.group()) if yr else 2024
        month = next((months[m] for m in months if m in txt.lower()), 1)

    # clean column names
    df.columns = [
        str(c).replace("\r", "").replace("\n", "").strip() for c in df.columns
    ]

    # detect city columns
    cities = [
        c
        for c in df.columns
        if c not in ["S. No.", "S.No.", "Description", "Unit"]
        and "Unnamed" not in str(c)
        and "Average" not in str(c)
        and "%" not in str(c)
        and pd.notna(c)
    ]

    # extract rows
    for _, row in df.iterrows():
        try:
            desc = row["Description"]
            if pd.isna(desc):
                continue
            for city in cities:
                try:
                    val = row[city]
                    if not pd.isna(val):
                        data.append(
                            {
                                "year": year,
                                "month": month,
                                "item": str(desc).strip(),
                                "city": str(city).replace("-", "").strip(),
                                "price": float(val),
                            }
                        )
                except:
                    pass
        except:
            pass

# sort data
df = (
    pd.DataFrame(data)
    .sort_values(["item", "city", "year", "month"])
    .reset_index(drop=True)
)

filled = []
# fill missing months (interpolation)
for (item, city), grp in df.groupby(["item", "city"]):
    grp = grp.reset_index(drop=True)
    for i in range(len(grp) - 1):
        filled.append(grp.iloc[i].to_dict())

        gap = (
            (grp.iloc[i + 1]["year"] - grp.iloc[i]["year"]) * 12
            + grp.iloc[i + 1]["month"]
            - grp.iloc[i]["month"]
        )

        if gap > 1:
            for j in range(1, gap):
                m = grp.iloc[i]["month"] + j
                y = grp.iloc[i]["year"]

                while m > 12:
                    m -= 12
                    y += 1

                # linear interpolation
                p = (
                    grp.iloc[i]["price"]
                    + (grp.iloc[i + 1]["price"] - grp.iloc[i]["price"]) * j / gap
                )

                filled.append(
                    {
                        "year": y,
                        "month": m,
                        "item": item,
                        "city": city,
                        "price": round(p, 2),
                    }
                )

    if len(grp) > 0:
        filled.append(grp.iloc[-1].to_dict())

# save final csv
pd.DataFrame(filled).sort_values(["year", "month", "item", "city"]).to_csv(
    "spi_data.csv", index=False
)
