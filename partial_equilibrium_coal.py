import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

total_coal = pd.read_csv("quantity_coal_imports_chn.csv")  # china's coal import data
coal_usa = pd.read_csv("dollar_coal_exports_usa.csv")  # us coal export data to china

total_coal_yearly = total_coal.groupby("Year")[
    "Value"
].sum()  # aggregate to yearly totals

proportion = (
    coal_usa.set_index("Year")["Value"] / total_coal_yearly
)  # calc us exports as % of china imports

total_coal["monthly_value_exports"] = (
    total_coal["Year"].map(proportion) * total_coal["Value"]
)  # distribute yearly proportion to months
total_coal["monthly_quantity_exports"] = (
    total_coal["Year"].map(proportion) * total_coal["Quantity"]
)  # calculate estimated us export quantity

total_coal["price_per_ton"] = (
    total_coal["monthly_value_exports"] / total_coal["monthly_quantity_exports"]
)  # derive unit price

x, y = total_coal[["monthly_quantity_exports"]], total_coal[["price_per_ton"]]

if not total_coal.empty:
    linear_model = LinearRegression().fit(x, y)  # fit linear model

    print(
        f"slope: {linear_model.coef_[0]}, intercept: {linear_model.intercept_}, r²: {linear_model.score(x, y)}"
    )

    plt.scatter(
        total_coal["monthly_quantity_exports"],
        y,
        color="b",
        label="Monthly Coal Exports",
    )
    plt.plot(
        total_coal["monthly_quantity_exports"],
        linear_model.predict(x),
        "r--",
        label="Linear Regression Line",
    )
    plt.ylabel("Price Per Ton (USD)")
    plt.xlabel("Quantity of Coal Imports (tons)")
    plt.title("Import Demand and Export Supply of Coal to China from the U.S.")
    plt.legend(loc="upper right")
    plt.grid()

p0 = total_coal["price_per_ton"].iloc[-1]  # initial equilibrium price

m0 = total_coal["monthly_quantity_exports"].iloc[-1]  # initial equilibrium quantity

plt.scatter(m0, p0, color="g", s=100, marker="*", label="Equilibrium Point")

if not total_coal.empty:
    slope = linear_model.coef_[0]

    x_min = total_coal["monthly_quantity_exports"].min() * 0.3
    x_max = total_coal["monthly_quantity_exports"].max()

    x_line = np.linspace(x_min, x_max)

    tariff = 0.15  # 15% retaliatory tariff by china

    y_line = slope * (x_line - m0) + p0  # import demand curve
    y_line2 = -slope * (x_line - m0) + p0  # export supply curve
    y_line3 = (-slope * (x_line - m0) + p0) * (1 + tariff)  # export supply with tariff

    plt.plot(x_line, y_line, "g-", label="Import Demand Curve")
    plt.plot(x_line, y_line2, "b-", label="Export Supply Curve")
    plt.plot(x_line, y_line3, "y-", label="Export Supply Curve (with tariff)")

    p1 = ((2 * (1 + tariff)) / (2 + tariff)) * p0  # new equilibrium price with tariff
    m1 = m0 + (tariff * p0) / (
        (2 + tariff) * slope
    )  # new equilibrium quantity with tariff

    plt.scatter(
        m1, p1, color="r", s=100, marker="s", label="Equilibirum Point (with tariff)"
    )

    plt.legend(loc="upper right")

region_a = m1 * (p1 - p0)  # increased price for consumers
region_b = 0.5 * (m0 - m1) * (p1 - p0)  # deadweight loss from decreased efficiency
region_c = (p0 - p1 / (1 + 0.25)) * (m1)  # government tariff revenue
region_d = (
    0.5 * (p0 - p1 / (1 + 0.25)) * (m0 - m1)
)  # deadweight loss from reduced trade

print(f"m0: {m0}, p0: {p0}, m1: {m1}, p1: {p1}")
print(f"region a: {region_a}")
print(f"region b: {region_b}")
print(f"region c: {region_c}")
print(f"region d: {region_d}")

plt.show()
