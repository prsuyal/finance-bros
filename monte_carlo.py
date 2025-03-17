import csv
import random

matrix = {}
with open("final_transition_matrix.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        current = row["current_state"].strip()
        nxt = row["next_state"].strip()
        prob = float(row["prob"])
        if current not in matrix:
            matrix[current] = []
        matrix[current].append((nxt, prob))


def update_tariff(current_state, current_tariff, opponent_tariff):
    new_tariff = current_tariff
    if current_state == "nothing":
        pass
    elif current_state == "titfortat":
        new_tariff = opponent_tariff
    elif current_state == "subretaliation":
        new_tariff = max(0.0, current_tariff - 0.02)
    elif current_state == "superretaliation":
        new_tariff = current_tariff + 0.02
    elif current_state == "subsidization":
        pass
    return new_tariff


p0_us = 104.0  # initial us price
m0_us = 567534.0  # initial us quantity
slope_us = -1.77579968e-05*1.1  # us demand curve slope
intercept_us = 114.34369908

p0_china = 109.0  # initial china price
m0_china = 1216199.0  # initial china quantity
slope_china = -1.54483697e-05  # china demand curve slope
intercept_china = 143.90003724


def compute_equilibrium_us(T):
    if (2.0 + T) == 0:
        return (p0_us, m0_us, 0, 0, 0, 0)

    p1 = (2.0 * (1.0 + T)) / (2.0 + T) * p0_us
    denom = (2.0 + T) * slope_us
    if abs(denom) < 1e-15:
        m1 = m0_us
    else:
        m1 = m0_us + (T * p0_us) / denom

    if (1.0 + T) == 0:
        return (p1, m1, 0, 0, 0, 0)

    A = m1 * (p1 - p0_us)
    B = 0.5 * (m0_us - m1) * (p1 - p0_us)
    C = (p0_us - (p1 / (1.0 + T))) * m0_us
    D = 0.5 * (p0_us - (p1 / (1.0 + T))) * (m0_us - m1)

    return (p1, m1, A, B, C, D)


def compute_equilibrium_china(T):
    if (2.0 + T) == 0:
        return (p0_china, m0_china, 0, 0, 0, 0)

    p1 = (2.0 * (1.0 + T)) / (2.0 + T) * p0_china
    denom = (2.0 + T) * slope_china
    if abs(denom) < 1e-15:
        m1 = m0_china
    else:
        m1 = m0_china + (T * p0_china) / denom

    if (1.0 + T) == 0:
        return (p1, m1, 0, 0, 0, 0)

    A = m1 * (p1 - p0_china)
    B = 0.5 * (m0_china - m1) * (p1 - p0_china)
    C = (p0_china - (p1 / (1.0 + T))) * m0_china
    D = 0.5 * (p0_china - (p1 / (1.0 + T))) * (m0_china - m1)
    return (p1, m1, A, B, C, D)


def compute_equilibrium_us_subsidy(S):
    A = m0_us * slope_us
    C = p0_us - m0_us * slope_us

    denom = 2.0 * slope_us
    if abs(denom) < 1e-15:
        m1 = m0_us
        p1 = p0_us
    else:
        m1 = m0_us - (S * p0_us) / denom
        p1 = p0_us * (1 - S / 2)

    gov = S * m1 * p0_us  # government
    prod = S * m1 * p1  # producer
    cons = 0.5 * (p0_us - p1) * (m1 - m0_us)  # consumer
    dwl = 0.5 * S * (m1 - m0_us) * p1  # deadweight loss

    return (p1, m1, gov, prod, cons, dwl)


def compute_equilibrium_china_subsidy(S):
    A = m0_china * slope_china
    B = p0_china - m0_china * slope_china

    denom = 2.0 * slope_china
    if abs(denom) < 1e-15:
        m1 = m0_china
        p1 = p0_china
    else:
        m1 = m0_china - (S * p0_china) / denom
        p1 = p0_china * (1 - S / 2)

    gov = S * m1 * p0_china
    prod = S * m1 * p1
    cons = 0.5 * (p0_china - p1) * (m1 - m0_china)
    dwl = 0.5 * S * (m1 - m0_china) * p1

    return (p1, m1, gov, prod, cons, dwl)


initial_state = "nothing"
num_steps = 12  # months in simulation
num_simulations = 100000  # monte carlo iterations

sumA_us = 0.0
sumB_us = 0.0
sumC_us = 0.0
sumD_us = 0.0

sumA_china = 0.0
sumB_china = 0.0
sumC_china = 0.0
sumD_china = 0.0

sum_gov_us = 0.0
sum_prod_us = 0.0
sum_cons_us = 0.0
sum_dwl_us = 0.0

sum_gov_china = 0.0
sum_prod_china = 0.0
sum_cons_china = 0.0
sum_dwl_china = 0.0

subsidy_count = 0

for _ in range(num_simulations):
    T_us = 0.10  # starting us tariff
    T_china = 0.05  # starting china tariff
    current_state = initial_state

    simA_us = simB_us = simC_us = simD_us = 0.0
    simA_china = simB_china = simC_china = simD_china = 0.0

    sim_subsidy_count = 0

    for _step in range(num_steps):
        transitions = matrix.get(current_state, [])
        r = random.random()
        cum = 0.0
        next_state = current_state
        for s, prob in transitions:
            cum += prob
            if r <= cum:
                next_state = s
                break

        T_us = update_tariff(next_state, T_us, T_china)
        T_china = update_tariff(next_state, T_china, T_us)

        if next_state == "subsidization":
            S_val = 0.02  # 2% subsidy
            p1_us, m1_us, gov_us, prod_us, cons_us, dwl_us = (
                compute_equilibrium_us_subsidy(S_val)
            )
            p1_cn, m1_cn, gov_cn, prod_cn, cons_cn, dwl_cn = (
                compute_equilibrium_china_subsidy(S_val)
            )

            sum_gov_us += gov_us
            sum_prod_us += prod_us
            sum_cons_us += cons_us
            sum_dwl_us += dwl_us

            sum_gov_china += gov_cn
            sum_prod_china += prod_cn
            sum_cons_china += cons_cn
            sum_dwl_china += dwl_cn

            sim_subsidy_count += 1

            A_us_val, B_us_val, C_us_val, D_us_val = prod_us, cons_us, gov_us, dwl_us
            A_ch_val, B_ch_val, C_ch_val, D_ch_val = prod_cn, cons_cn, gov_cn, dwl_cn
        else:
            p1_us, m1_us, A_us_val, B_us_val, C_us_val, D_us_val = (
                compute_equilibrium_us(T_us)
            )
            p1_cn, m1_cn, A_ch_val, B_ch_val, C_ch_val, D_ch_val = (
                compute_equilibrium_china(T_china)
            )

        simA_us += A_us_val
        simB_us += B_us_val
        simC_us += C_us_val
        simD_us += D_us_val

        simA_china += A_ch_val
        simB_china += B_ch_val
        simC_china += C_ch_val
        simD_china += D_ch_val

        current_state = next_state

    subsidy_count += sim_subsidy_count

    sumA_us += simA_us
    sumB_us += simB_us
    sumC_us += simC_us
    sumD_us += simD_us

    sumA_china += simA_china
    sumB_china += simB_china
    sumC_china += simC_china
    sumD_china += simD_china

avgA_us = sumA_us / num_simulations
avgB_us = sumB_us / num_simulations
avgC_us = sumC_us / num_simulations
avgD_us = sumD_us / num_simulations
avg_total_us = avgA_us + avgB_us + avgC_us + avgD_us

avgA_china = sumA_china / num_simulations
avgB_china = sumB_china / num_simulations
avgC_china = sumC_china / num_simulations
avgD_china = sumD_china / num_simulations
avg_total_china = avgA_china + avgB_china + avgC_china + avgD_china

avg_subsidy_count = subsidy_count / num_simulations
if subsidy_count > 0:
    avg_gov_us = sum_gov_us / subsidy_count
    avg_prod_us = sum_prod_us / subsidy_count
    avg_cons_us = sum_cons_us / subsidy_count
    avg_dwl_us = sum_dwl_us / subsidy_count

    avg_gov_china = sum_gov_china / subsidy_count
    avg_prod_china = sum_prod_china / subsidy_count
    avg_cons_china = sum_cons_china / subsidy_count
    avg_dwl_china = sum_dwl_china / subsidy_count
else:
    avg_gov_us = avg_prod_us = avg_cons_us = avg_dwl_us = 0
    avg_gov_china = avg_prod_china = avg_cons_china = avg_dwl_china = 0

print("avgs for us semis (over 12 steps, 100k sims)")
print(f"region a us: {avgA_us:,.2f}")
print(f"region b us: {avgB_us:,.2f}")
print(f"region c us: {avgC_us:,.2f}")
print(f"region d us: {avgD_us:,.2f}")
print(f"total: {avg_total_us:,.2f}")

print("\navgs for china coal (over 12 steps, 100k sims)")
print(f"region a china: {avgA_china:,.2f}")
print(f"region b china: {avgB_china:,.2f}")
print(f"region c china: {avgC_china:,.2f}")
print(f"region d china: {avgD_china:,.2f}")
print(f"total: {avg_total_china:,.2f}")

print("\nsubsidization effects")
print(f"avg subsidy instances per sim: {avg_subsidy_count:.4f}")
if subsidy_count > 0:
    print("\nus subsidy effects per instance:")
    print(f"gov cost: {avg_gov_us:,.2f}")
    print(f"producer surplus: {avg_prod_us:,.2f}")
    print(f"consumer surplus: {avg_cons_us:,.2f}")
    print(f"deadweight loss: {avg_dwl_us:,.2f}")

    print("\nchina subsidy effects per instance:")
    print(f"gov cost: {avg_gov_china:,.2f}")
    print(f"producer surplus: {avg_prod_china:,.2f}")
    print(f"consumer surplus: {avg_cons_china:,.2f}")
    print(f"deadweight loss: {avg_dwl_china:,.2f}")

print("\ndone!")
