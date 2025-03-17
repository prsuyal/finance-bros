import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

total_imports = pd.read_csv("total_imports.csv")
semi_imports = pd.read_csv("semi_imports.csv")
ipi = pd.read_csv("ipi.csv")  # import price index data

month_map = {
    "Jan": "M01",
    "Feb": "M02",
    "Mar": "M03",
    "Apr": "M04",
    "May": "M05",
    "Jun": "M06",
    "Jul": "M07",
    "Aug": "M08",
    "Sep": "M09",
    "Oct": "M10",
    "Nov": "M11",
    "Dec": "M12",
}

total_imports["year month"] = (
    total_imports["Year"].astype(str) + "-" + total_imports["Month"].map(month_map)
)  # standardize date format for merging

total_imports_yearly = total_imports.groupby("Year")["Value"].sum()  # aggregate to yearly totals

proportion = semi_imports.set_index("Year")["Value"] / total_imports_yearly  # calc semi imports as % of total

total_imports["estimated_semi_imports"] = (
    total_imports["Year"].map(proportion) * total_imports["Value"]
)  # distribute yearly proportion to months

ipi["year month"] = ipi["Year"].astype(str) + "-" + ipi["Period"]
ipi = ipi[["year month", "Value"]].rename(columns={"Value": "ipi"})

df = total_imports.merge(ipi, on="year month", how="inner").dropna()  # combine datasets

df["semi_imports_per_ipi"] = df["estimated_semi_imports"] / df["ipi"]  # normalize by price (quantity proxy)

x, y = df[["semi_imports_per_ipi"]], df[["ipi"]]

if not df.empty:
    linear_model = LinearRegression().fit(x, y) # fit linear model

    print(
        f"slope: {linear_model.coef_[0]}, intercept: {linear_model.intercept_}, r²: {linear_model.score(x, y)}"
    )

    plt.scatter(
        df["semi_imports_per_ipi"], 
        y,
        color="b",
        label="Monthly Semiconductor Imports"
    )
    plt.plot(
        df["semi_imports_per_ipi"],
        linear_model.predict(x),
        "r--",
        label="Linear Regression Line"
    )
    plt.ylabel("Price Index (IPI)")
    plt.xlabel("Value of Semiconductor Imports / IPI")
    plt.title("Import Demand and Export Supply of Semiconductors to U.S.")
    plt.legend(loc='upper right')
    plt.grid()

p0 = ipi["ipi"].iloc[-1]  # initial equilibrium price
print(f"initial price {p0}")

m0 = total_imports["estimated_semi_imports"].iloc[-1] / ipi["ipi"].iloc[-1]  # initial equilibrium quantity
print(f"initial import {m0}")

plt.scatter(m0, p0, color="g", s=100, marker="*", label="Equilibrium Point")

if not df.empty:
    slope = linear_model.coef_[0]

    x_min = df["semi_imports_per_ipi"].min() * 0.3
    x_max = df["semi_imports_per_ipi"].max()

    x_line = np.linspace(x_min, x_max)

    tariff = 0.1  # 10% tariff imposed by president trump

    slope = slope*1.1

    y_line = slope * (x_line - m0) + p0  # import demand 
    y_line2 = -slope * (x_line - m0) + p0  # export supply 
    y_line3 = (-slope * (x_line - m0) + p0) * (1 + tariff)  # export supply with tariff

    plt.plot(x_line, y_line, "g-", label="Import Demand Curve")
    plt.plot(x_line, y_line2, "b-", label="Export Supply Curve")
    plt.plot(x_line, y_line3, "y-", label="Export Supply Curve (with tariff)")

    p1 = ((2 * (1 + tariff)) / (2 + tariff)) * p0  # new equilibrium price with tariff
    m1 = m0 + (tariff * p0) / ((2 + tariff) * slope)  # new equilibrium quantity with tariff

    plt.scatter(m1, p1, color="r", s=100, marker="s", label="Equilibrium Point (with tariff)")

    plt.legend(loc='upper right')

region_a = m1 * (p1 - p0)  # increased price for consumers
region_b = 0.5 * (m0 - m1) * (p1 - p0)  # deadweight loss from decreased efficiency
region_c = (p0 - p1 / (1 + 0.25)) * (m1)  # domestic government tariff revenue
region_d = 0.5 * (p0 - p1 / (1 + 0.25)) * (m0 - m1)  # foreign government deadweight loss from reduced exports

print(f"m0: {m0}, p0: {p0}, m1: {m1}, p1: {p1}")
print(f"region a: {region_a}")
print(f"region b: {region_b}")
print(f"region c: {region_c}")
print(f"region d: {region_d}")

plt.show()
