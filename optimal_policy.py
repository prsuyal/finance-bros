import csv
import random
import heapq
import pandas as pd
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from econml.dml import CausalForestDML

# load transition matrix for china's behavior
china_matrix = {}
with open("final_transition_matrix.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        current = row["current_state"].strip()
        nxt = row["next_state"].strip()
        nxt = row["next_state"].strip()
        prob = float(row["prob"])
        if current not in china_matrix:
            china_matrix[current] = []
        china_matrix[current].append((nxt, prob))


def update_tariff(current_state, current_tariff, opponent_tariff):
    new_tariff = current_tariff
    if current_state == "nothing":
        pass
    elif current_state == "titfortat":
        new_tariff = opponent_tariff
    elif current_state == "subretaliation":
        new_tariff = max(0.0, current_tariff - 0.02)  # decrease by 2%
    elif current_state == "superretaliation":
        new_tariff = current_tariff + 0.02  # increase by 2%
    elif current_state == "subsidization":
        pass
    return new_tariff


# initial market parameters, same as before
p0_us = 104.0
m0_us = 567534.0
slope_us = -1.77579968e-05
intercept_us = 114.34369908

p0_china = 109.0
m0_china = 1216199.0
slope_china = -1.54483697e-05
intercept_china = 143.90003724


def compute_equilibrium_us(T):
    if (2.0 + T) == 0:
        return (p0_us, m0_us, 0, 0, 0, 0)

    p1 = (2.0 * (1.0 + T)) / (2.0 + T) * p0_us  # new equilibrium price
    denom = (2.0 + T) * slope_us
    if abs(denom) < 1e-15:
        m1 = m0_us
    else:
        m1 = m0_us + (T * p0_us) / denom  # new equilibrium quantity
    if (1.0 + T) == 0:
        return (p1, m1, 0, 0, 0, 0)

    # welfare regions
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
    denom = 2.0 * slope_us
    if abs(denom) < 1e-15:
        m1 = m0_us
        p1 = p0_us
    else:
        m1 = m0_us - (S * p0_us) / denom
        p1 = p0_us * (1 - S / 2)

    # subsidy welfare components
    gov = S * m1 * p0_us
    prod = S * m1 * p1
    cons = 0.5 * (p0_us - p1) * (m1 - m0_us)
    dwl = 0.5 * S * (m1 - m0_us) * p1
    return (p1, m1, gov, prod, cons, dwl)


def compute_equilibrium_china_subsidy(S):
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


us_policies = [
    "nothing",
    "titfortat",
    "subretaliation",
    "superretaliation",
    "subsidization",
]


def random_us_policy_action():
    # select random policy with appropriate magnitude
    policy = random.choice(us_policies)
    if policy == "nothing":
        return (policy, 0.0)
    elif policy == "subretaliation":
        mag = random.uniform(0.0, 0.05)  # random decrease up to 5%
        return (policy, mag)
    elif policy == "superretaliation":
        mag = random.uniform(0.0, 0.05)  # random increase up to 5%
        return (policy, mag)
    elif policy == "titfortat":
        return (policy, 0.0)
    elif policy == "subsidization":
        mag = random.uniform(0.0, 0.1)  # random subsidy up to 10%
        return (policy, mag)


def simulate_random_policy_sequence(num_steps=12):
    # simulate trade policy interaction over time
    china_current_state = "nothing"
    T_china = 0.05
    T_us = 0.10

    us_total_score = 0.0
    us_policy_sequence = []

    for step in range(num_steps):
        # china follows markov process for state transitions
        transitions = china_matrix.get(china_current_state, [])
        r = random.random()
        c = 0.0
        china_next_state = china_current_state
        for s, prob in transitions:
            c += prob
            if r <= c:
                china_next_state = s
                break

        # apply china's policy based on state
        if china_next_state == "subsidization":
            S_china = random.uniform(0, 0.1)  # random subsidy magnitude
            subsidy_info_china = compute_equilibrium_china_subsidy(S_china)
            china_mode = "subsidy"
            china_subsidy_val = S_china
        else:
            T_china = update_tariff(china_next_state, T_china, T_us)
            china_mode = "tariff"
            china_subsidy_val = None
        china_current_state = china_next_state

        us_policy, mag = random_us_policy_action()

        # apply us policy and calculate welfare effects
        if us_policy == "subsidization":
            S_us = mag
            p1_us, m1_us, gov_us, prod_us, cons_us, dwl_us = (
                compute_equilibrium_us_subsidy(S_us)
            )
            step_score_us = prod_us + cons_us + dwl_us
            us_mode = "subsidy"
        else:
            # apply tariff policy
            if us_policy == "nothing":
                pass
            elif us_policy == "subretaliation":
                T_us = max(0.0, T_us - mag)
            elif us_policy == "superretaliation":
                T_us = min(0.2, T_us + mag)
            elif us_policy == "titfortat":
                T_us = T_china
            us_mode = "tariff"
            _, _, A_us, B_us, C_us, D_us = compute_equilibrium_us(T_us)
            step_score_us = A_us + B_us + C_us + D_us

        us_total_score += step_score_us  # accumulate welfare over time

        # record what happened this step
        us_policy_sequence.append(
            (
                us_policy,
                mag,
                T_us,
                china_next_state,
                T_china,
                us_mode,
                china_mode,
                china_subsidy_val,
            )
        )

    return (us_total_score, us_policy_sequence)


def extract_features(seq, score):
    # extract policy features from a sequence for causal analysis
    features = {}
    features["score"] = score
    features["final_T_us"] = seq[-1][2]  # final us tariff
    features["final_T_china"] = seq[-1][4]  # final china tariff

    # count policy types used
    features["count_nothing"] = sum(1 for step in seq if step[0] == "nothing")
    features["count_titfortat"] = sum(1 for step in seq if step[0] == "titfortat")
    features["count_subretaliation"] = sum(
        1 for step in seq if step[0] == "subretaliation"
    )
    features["count_superretaliation"] = sum(
        1 for step in seq if step[0] == "superretaliation"
    )
    features["count_subsidization"] = sum(
        1 for step in seq if step[0] == "subsidization"
    )

    # calculate average magnitudes when used
    us_subsidy_mags = [step[1] for step in seq if step[0] == "subsidization"]
    features["avg_us_subsidy_mag"] = (
        sum(us_subsidy_mags) / len(us_subsidy_mags) if len(us_subsidy_mags) > 0 else 0.0
    )
    us_subret_mags = [step[1] for step in seq if step[0] == "subretaliation"]
    features["avg_us_subret_mag"] = (
        sum(us_subret_mags) / len(us_subret_mags) if len(us_subret_mags) > 0 else 0.0
    )
    us_superret_mags = [step[1] for step in seq if step[0] == "superretaliation"]
    features["avg_us_superret_mag"] = (
        sum(us_superret_mags) / len(us_superret_mags)
        if len(us_superret_mags) > 0
        else 0.0
    )

    # define treatment variables for causal analysis
    features["treatment_subsidization_strength"] = features["avg_us_subsidy_mag"]
    features["treatment_subretaliation_strength"] = features["avg_us_subret_mag"]
    features["treatment_superretaliation_strength"] = features["avg_us_superret_mag"]
    features["avg_T_us"] = sum(step[2] for step in seq) / len(seq)
    features["avg_T_china"] = sum(step[4] for step in seq) / len(seq)
    features["treatment_nothing_strength"] = (
        features["count_nothing"] / 12.0
    )  # as proportion
    features["treatment_titfortat_strength"] = (
        features["count_titfortat"] / 12.0
    )  # as proportion
    return features


def run_causal_ml_analysis(sample_size):
    # causal ML analysis to understand policy effects on welfare
    print(
        f"running continuous-treatment causal analysis with sample size: {sample_size}"
    )

    # generate data by simulating many policy sequences
    features_list = []
    for i in range(sample_size):
        score, seq = simulate_random_policy_sequence(12)
        feat = extract_features(seq, score)
        features_list.append(feat)

    df = pd.DataFrame(features_list).fillna(0)

    print("\ndata summary after preprocessing:")
    print(df.describe())

    # define which treatment variables to analyze
    treatment_vars = [
        "treatment_subsidization_strength",  # subsidy magnitude
        "treatment_subretaliation_strength",  # tariff reduction magnitude
        "treatment_superretaliation_strength",  # tariff increase magnitude
        "treatment_nothing_strength",  # proportion of doing nothing
        "treatment_titfortat_strength",  # proportion of matching china
    ]

    outcome_col = "score"  # welfare score is our outcome

    # analyze each treatment separately
    for treat_var in treatment_vars:
        print(f"analyzing T = '{treat_var}' as continuous treatment")

        y = df[outcome_col].values  # outcome
        T = df[treat_var].values  # treatment

        # control vars
        covar_cols = [c for c in df.columns if c not in [outcome_col, treat_var]]
        X = df[covar_cols].values

        print(
            f"treatment stats for {treat_var}: min={T.min():.4f}, max={T.max():.4f}, mean={T.mean():.4f}"
        )
        print(
            f"outcome stats: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}"
        )

        # skip if treatment has no variation
        if np.var(T) < 1e-12:
            print(f"WARNING: {treat_var} variance is ~0; skipping analysis")
            continue

        # split data for training + testing
        X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
            X, y, T, test_size=0.2, random_state=42
        )

        # causal forest for heterogeneous treatment effect estimation
        est = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=50, min_samples_leaf=5, max_depth=10, random_state=42
            ),
            model_t=RandomForestRegressor(
                n_estimators=50, min_samples_leaf=5, max_depth=10, random_state=42
            ),
            cv=3,
            random_state=42,
        )

        est.fit(y_train, T_train, X=X_train)
        print("CausalForestDML fit completed")

        # estimate treatment effect going from no treatment to moderate treatment
        T0, T1 = 0.0, 0.05  # compare effect of increasing treatment from 0 to 5%
        effect_predictions = est.effect(X_test, T0=T0, T1=T1)

        # calculate average treatment effect and confidence intervals
        ate_estimate = effect_predictions.mean()
        ate_ci_lower = np.percentile(effect_predictions, 2.5)
        ate_ci_upper = np.percentile(effect_predictions, 97.5)

        print(f"ATE for T going from {T0} to {T1}: {ate_estimate:.4f}")
        print(f"approx 95% CI: [{ate_ci_lower:.4f}, {ate_ci_upper:.4f}]")


def main():
    import sys

    run_policy_simulation = True
    run_causal_analysis = True

    if run_policy_simulation:
        NUM_TRIALS = 10_000_000
        NUM_STEPS = 12
        TOP_N = 5

        best_heap = []  # min-heap to track top performers

        for i in range(NUM_TRIALS):
            score, seq = simulate_random_policy_sequence(NUM_STEPS)
            if len(best_heap) < TOP_N:
                best_heap.append((score, seq))
                if len(best_heap) == TOP_N:
                    heapq.heapify(best_heap)
            else:
                if score > best_heap[0][0]:
                    heapq.heapreplace(best_heap, (score, seq))
                if (i + 1) % 100000 == 0:
                    print(f"completed {i+1} / {NUM_TRIALS} trials", file=sys.stderr)

        best_heap.sort(key=lambda x: x[0], reverse=True)  # sort best to worst

        # display top performing policy sequences
        print(f"\ntop {TOP_N} us policy sequences out of {NUM_TRIALS} trials")
        for rank, (score, policy_seq) in enumerate(best_heap, start=1):
            print(f"{rank}: us total score = {score:,.2f}")
            print(
                " step-by-step (us policy, us subsidy mag, us tariff, china state, china tariff, us mode, china mode, china subsidy):"
            )
            for stepinfo in policy_seq:
                print(f" {stepinfo}")
            print("")

    if run_causal_analysis:
        run_causal_ml_analysis(sample_size=10_000_000)


if __name__ == "__main__":
    main()
