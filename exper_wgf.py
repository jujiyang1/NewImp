import gc
import pandas as pd
import os
import argparse
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

from utils.utils import enable_reproducible_results, simulate_scenarios, overall_MAE, overall_RMSE, calculate_wasserstein_distance
from model.wgf_imp import NeuralGradFlowImputer
import torch
torch.set_default_tensor_type('torch.DoubleTensor')


# basic params
parser = argparse.ArgumentParser(prog='Basic')
parser.add_argument('--model', default='WassGF')
parser.add_argument('--seed', default=4, type=int)
parser.add_argument('--outpath', default='./results/flowImpResults')
parser.add_argument('--verbose', default=1)

parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--n_epochs', default=150, type=int)
parser.add_argument('--n_pairs', default=10, type=int)
parser.add_argument('--noise', default=1e-4, type=float)
parser.add_argument('--numItermax', default=1000, type=int)
parser.add_argument('--stopThr', default=1e-3, type=float)
parser.add_argument('--alpha', default=0, type=float)

parser.add_argument('--bandwidth', default=0.5, type=float)
parser.add_argument('--entropy_reg', default=10.0, type=float)
parser.add_argument('--score_net_epoch', default=200, type=int)
parser.add_argument('--iter_time', default=2, type=int)


parser.add_argument('--initializer', default='mean', type=str)
parser.add_argument('--mlp_hidden', default='[128, 128]', type=str)

parser.add_argument('--ode_step', default=1.0e-1, type=float)
parser.add_argument('--score_lr', default=1.0e-3, type=float)
parser.add_argument('--dataset_name', default="blood_transfusion", type=str)


# parse_args operations
args = parser.parse_args()


# get the dataset
if not os.path.exists("./datasets"):
    os.makedirs("./datasets")

ground_truth = pd.read_csv("normalized_data.csv").values







SCENARIO = [
    "MAR",
    "MNAR",
    # "MCAR" # optional
]
P_MISS = [
    0.3,
]

# Number of experiment runs
NUM_RUNS = 10

# Initialize dictionaries to store results from all runs
all_runs_results = {
    method: {
        metric: {
            scenario: {
                p_miss: [] for p_miss in P_MISS
            } for scenario in SCENARIO
        } for metric in ['missing_mae', 'overall_mae', 'overall_rmse', 'wass']
    } for method in ['NeuralGradFlow', 'MICE', 'RFMICE']
}

# Initialize DataFrame to store all results
result_df = pd.DataFrame()

# Main experiment loop - run experiments NUM_RUNS times
for run in range(NUM_RUNS):
    print(f"\n{'='*40}")
    print(f"STARTING EXPERIMENT RUN {run+1}/{NUM_RUNS}")
    print(f"{'='*40}\n")

    # Setup random seed for this run (use different seed for each run)
    run_seed = args.seed + run
    enable_reproducible_results(run_seed)
    print(f"Running with seed: {run_seed}")

    X = ground_truth
    # Generate new missing values for each run
    imputation_scenarios = simulate_scenarios(X, mechanisms=SCENARIO, percentages=P_MISS)

    # Store results for this run
    run_result_df = pd.DataFrame()
    for scenario in SCENARIO:
        for p_miss in P_MISS:
            # 获取数据
            x, x_miss, mask = imputation_scenarios[scenario][p_miss]
            mask_np = mask.values.astype(bool)

            # 1. NeuralGradFlowImputer
            print(f"\n===== Running NeuralGradFlowImputer for {scenario} with {p_miss*100}% missing data =====")
            model = NeuralGradFlowImputer(entropy_reg=args.entropy_reg, bandwidth=args.bandwidth,
                                          score_net_epoch=args.score_net_epoch, niter=args.iter_time,
                                          initializer=None, mlp_hidden=eval(args.mlp_hidden), lr=args.ode_step,
                                          score_net_lr=args.score_lr)

            enable_reproducible_results(run_seed)
            x_imputed_ngf, result_list = model.fit_transform(x_miss.copy().values, verbose=True, report_interval=1, X_true=x.copy().values)

            # Calculate overall MAE and RMSE for NeuralGradFlowImputer
            ngf_overall_mae = overall_MAE(x_imputed_ngf.detach().cpu().numpy(), x.copy().values)
            ngf_overall_rmse = overall_RMSE(x_imputed_ngf.detach().cpu().numpy(), x.copy().values)

            # Calculate missing-only MAE and RMSE for NeuralGradFlowImputer
            ngf_missing_mae = np.mean(np.abs(x_imputed_ngf.detach().cpu().numpy()[mask_np] - x.copy().values[mask_np]))
            ngf_missing_rmse = np.sqrt(np.mean(np.square(x_imputed_ngf.detach().cpu().numpy()[mask_np] - x.copy().values[mask_np])))

            # Calculate Wasserstein distance for NeuralGradFlowImputer
            ngf_wass = calculate_wasserstein_distance(
                x_imputed_ngf.detach().cpu().numpy(),
                x.copy().values,
                mask.values
            )

            # Store results for this run
            all_runs_results['NeuralGradFlow']['missing_mae'][scenario][p_miss].append(ngf_missing_mae)
            all_runs_results['NeuralGradFlow']['overall_mae'][scenario][p_miss].append(ngf_overall_mae)
            all_runs_results['NeuralGradFlow']['overall_rmse'][scenario][p_miss].append(ngf_overall_rmse)
            all_runs_results['NeuralGradFlow']['wass'][scenario][p_miss].append(ngf_wass)

            print(f"NeuralGradFlowImputer - Overall MAE: {ngf_overall_mae:.6f}")
            print(f"NeuralGradFlowImputer - Overall RMSE: {ngf_overall_rmse:.6f}")
            print(f"NeuralGradFlowImputer - Missing-only MAE: {ngf_missing_mae:.6f}")
            print(f"NeuralGradFlowImputer - Wasserstein distance: {ngf_wass:.6f}")
            print("-" * 60)

            # 2. MICE (Multiple Imputation by Chained Equations)
            print(f"\n===== Running MICE for {scenario} with {p_miss*100}% missing data =====")
            enable_reproducible_results(run_seed)
            mice_imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=100,
                random_state=run_seed,
                verbose=0
            )
            x_miss_np = x_miss.copy().values
            x_imputed_mice = mice_imputer.fit_transform(x_miss_np)

            # Calculate overall MAE and RMSE for MICE
            mice_overall_mae = overall_MAE(x_imputed_mice, x.copy().values)
            mice_overall_rmse = overall_RMSE(x_imputed_mice, x.copy().values)

            # Calculate missing-only MAE and RMSE for MICE
            mice_missing_mae = np.mean(np.abs(x_imputed_mice[mask_np] - x.copy().values[mask_np]))
            mice_missing_rmse = np.sqrt(np.mean(np.square(x_imputed_mice[mask_np] - x.copy().values[mask_np])))

            # Calculate Wasserstein distance for MICE
            mice_wass = calculate_wasserstein_distance(
                x_imputed_mice,
                x.copy().values,
                mask.values
            )

            # Store results for this run
            all_runs_results['MICE']['missing_mae'][scenario][p_miss].append(mice_missing_mae)
            all_runs_results['MICE']['overall_mae'][scenario][p_miss].append(mice_overall_mae)
            all_runs_results['MICE']['overall_rmse'][scenario][p_miss].append(mice_overall_rmse)
            all_runs_results['MICE']['wass'][scenario][p_miss].append(mice_wass)

            print(f"MICE - Overall MAE: {mice_overall_mae:.6f}")
            print(f"MICE - Overall RMSE: {mice_overall_rmse:.6f}")
            print(f"MICE - Missing-only MAE: {mice_missing_mae:.6f}")
            print(f"MICE - Wasserstein distance: {mice_wass:.6f}")
            print("-" * 60)

            # 3. RFMICE (Random Forest MICE)
            print(f"\n===== Running RFMICE for {scenario} with {p_miss*100}% missing data =====")
            enable_reproducible_results(run_seed)
            rfmice_imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=run_seed),
                max_iter=10,
                random_state=run_seed,
                verbose=0
            )
            x_miss_np = x_miss.copy().values
            x_imputed_rfmice = rfmice_imputer.fit_transform(x_miss_np)

            # Calculate overall MAE and RMSE for RFMICE
            rfmice_overall_mae = overall_MAE(x_imputed_rfmice, x.copy().values)
            rfmice_overall_rmse = overall_RMSE(x_imputed_rfmice, x.copy().values)

            # Calculate missing-only MAE and RMSE for RFMICE
            rfmice_missing_mae = np.mean(np.abs(x_imputed_rfmice[mask_np] - x.copy().values[mask_np]))
            rfmice_missing_rmse = np.sqrt(np.mean(np.square(x_imputed_rfmice[mask_np] - x.copy().values[mask_np])))

            # Calculate Wasserstein distance for RFMICE
            rfmice_wass = calculate_wasserstein_distance(
                x_imputed_rfmice,
                x.copy().values,
                mask.values
            )

            # Store results for this run
            all_runs_results['RFMICE']['missing_mae'][scenario][p_miss].append(rfmice_missing_mae)
            all_runs_results['RFMICE']['overall_mae'][scenario][p_miss].append(rfmice_overall_mae)
            all_runs_results['RFMICE']['overall_rmse'][scenario][p_miss].append(rfmice_overall_rmse)
            all_runs_results['RFMICE']['wass'][scenario][p_miss].append(rfmice_wass)

            print(f"RFMICE - Overall MAE: {rfmice_overall_mae:.6f}")
            print(f"RFMICE - Overall RMSE: {rfmice_overall_rmse:.6f}")
            print(f"RFMICE - Missing-only MAE: {rfmice_missing_mae:.6f}")
            print(f"RFMICE - Wasserstein distance: {rfmice_wass:.6f}")
            print("-" * 60)

            # 比较三种方法
            print(f"\n===== Comparison of Methods for {scenario} with {p_miss*100}% missing data =====")
            print("Method          | Missing-only MAE | Wasserstein Dist | Overall MAE    | Overall RMSE")
            print("-" * 80)
            print(f"NeuralGradFlow  | {ngf_missing_mae:.6f}      | {ngf_wass:.6f}       | {ngf_overall_mae:.6f}   | {ngf_overall_rmse:.6f}")
            print(f"MICE            | {mice_missing_mae:.6f}      | {mice_wass:.6f}       | {mice_overall_mae:.6f}   | {mice_overall_rmse:.6f}")
            print(f"RFMICE          | {rfmice_missing_mae:.6f}      | {rfmice_wass:.6f}       | {rfmice_overall_mae:.6f}   | {rfmice_overall_rmse:.6f}")
            print("=" * 80)

            # 确定最佳方法
            methods = ["NeuralGradFlow", "MICE", "RFMICE"]
            missing_maes = [ngf_missing_mae, mice_missing_mae, rfmice_missing_mae]
            wasserstein_dists = [ngf_wass, mice_wass, rfmice_wass]
            overall_maes = [ngf_overall_mae, mice_overall_mae, rfmice_overall_mae]
            overall_rmses = [ngf_overall_rmse, mice_overall_rmse, rfmice_overall_rmse]

            best_missing_mae_idx = np.argmin(missing_maes)
            best_wasserstein_idx = np.argmin(wasserstein_dists)
            best_overall_mae_idx = np.argmin(overall_maes)
            best_overall_rmse_idx = np.argmin(overall_rmses)

            print(f"Best method by Missing-only MAE: {methods[best_missing_mae_idx]}")
            print(f"Best method by Wasserstein distance: {methods[best_wasserstein_idx]}")
            print(f"Best method by Overall MAE: {methods[best_overall_mae_idx]}")
            print(f"Best method by Overall RMSE: {methods[best_overall_rmse_idx]}")

            # Add overall metrics to the result list for NeuralGradFlowImputer
            for dict_idx in result_list:
                dict_idx["overall_mae"] = ngf_overall_mae
                dict_idx["overall_rmse"] = ngf_overall_rmse
                dict_idx["mice_missing_mae"] = mice_missing_mae
                dict_idx["mice_wass"] = mice_wass
                dict_idx["mice_overall_mae"] = mice_overall_mae
                dict_idx["mice_overall_rmse"] = mice_overall_rmse
                dict_idx["rfmice_missing_mae"] = rfmice_missing_mae
                dict_idx["rfmice_wass"] = rfmice_wass
                dict_idx["rfmice_overall_mae"] = rfmice_overall_mae
                dict_idx["rfmice_overall_rmse"] = rfmice_overall_rmse
                dict_idx["best_method_missing_mae"] = methods[best_missing_mae_idx]
                dict_idx["best_method_wasserstein"] = methods[best_wasserstein_idx]
                dict_idx["best_method_overall_mae"] = methods[best_overall_mae_idx]
                dict_idx["best_method_overall_rmse"] = methods[best_overall_rmse_idx]
                dict_idx["run"] = run + 1  # Add run number to track results from different runs

            # print(result_list)
            result_list = [pd.DataFrame(dict_idx, index=[0]) for dict_idx in result_list]
            temp_result_df = pd.concat(result_list, axis=0)
            temp_result_df["missing"] = scenario
            temp_result_df["dataset_name"] = args.dataset_name
            temp_result_df["seed"] = run_seed
            temp_result_df["p_miss"] = p_miss
            run_result_df = pd.concat([run_result_df, temp_result_df], axis=0)
            del x, x_miss, mask, model
            gc.collect()
            torch.cuda.empty_cache()

    # Add this run's results to the overall results DataFrame
    result_df = pd.concat([result_df, run_result_df], axis=0)


# Filter results for the last interval
result_df = result_df[result_df["interval"] == int(args.iter_time - 1)].reset_index(drop=True)

# Create directory for results if it doesn't exist
os.makedirs(args.outpath) if not os.path.exists(args.outpath) else None

# Update CSV filename to indicate multiple runs
csv_name = (f"model_{args.model}_data_{args.dataset_name}_seed_{args.seed}_runs_{NUM_RUNS}"
            f"_ode_step_{args.lr}_bandwidth_{args.bandwidth}_reg_{args.entropy_reg}"
            f"_score_{eval(args.mlp_hidden)[0]}_epoch_{args.score_net_epoch}.csv")

# Calculate average metrics across all runs
print("\n" + "="*80)
print(f"===== SUMMARY OF RESULTS ACROSS {NUM_RUNS} RUNS =====")
print("="*80)

for scenario in SCENARIO:
    for p_miss in P_MISS:
        # Get all results for this scenario and missing percentage
        scenario_df = result_df[(result_df["missing"] == scenario) & (result_df["p_miss"] == p_miss) & (result_df["interval"] == int(args.iter_time - 1))]

        if not scenario_df.empty:
            print(f"\nScenario: {scenario}, Missing: {p_miss*100}%")

            # Calculate average metrics for each method
            ngf_avg_missing_mae = np.mean(all_runs_results['NeuralGradFlow']['missing_mae'][scenario][p_miss])
            ngf_avg_overall_mae = np.mean(all_runs_results['NeuralGradFlow']['overall_mae'][scenario][p_miss])
            ngf_avg_overall_rmse = np.mean(all_runs_results['NeuralGradFlow']['overall_rmse'][scenario][p_miss])
            ngf_avg_wass = np.mean(all_runs_results['NeuralGradFlow']['wass'][scenario][p_miss])

            mice_avg_missing_mae = np.mean(all_runs_results['MICE']['missing_mae'][scenario][p_miss])
            mice_avg_overall_mae = np.mean(all_runs_results['MICE']['overall_mae'][scenario][p_miss])
            mice_avg_overall_rmse = np.mean(all_runs_results['MICE']['overall_rmse'][scenario][p_miss])
            mice_avg_wass = np.mean(all_runs_results['MICE']['wass'][scenario][p_miss])

            rfmice_avg_missing_mae = np.mean(all_runs_results['RFMICE']['missing_mae'][scenario][p_miss])
            rfmice_avg_overall_mae = np.mean(all_runs_results['RFMICE']['overall_mae'][scenario][p_miss])
            rfmice_avg_overall_rmse = np.mean(all_runs_results['RFMICE']['overall_rmse'][scenario][p_miss])
            rfmice_avg_wass = np.mean(all_runs_results['RFMICE']['wass'][scenario][p_miss])

            # Calculate standard deviations for each method
            ngf_std_missing_mae = np.std(all_runs_results['NeuralGradFlow']['missing_mae'][scenario][p_miss])
            mice_std_missing_mae = np.std(all_runs_results['MICE']['missing_mae'][scenario][p_miss])
            rfmice_std_missing_mae = np.std(all_runs_results['RFMICE']['missing_mae'][scenario][p_miss])

            # Determine best method based on average metrics
            methods = ["NeuralGradFlow", "MICE", "RFMICE"]
            avg_missing_maes = [ngf_avg_missing_mae, mice_avg_missing_mae, rfmice_avg_missing_mae]
            avg_wasserstein_dists = [ngf_avg_wass, mice_avg_wass, rfmice_avg_wass]
            avg_overall_maes = [ngf_avg_overall_mae, mice_avg_overall_mae, rfmice_avg_overall_mae]
            avg_overall_rmses = [ngf_avg_overall_rmse, mice_avg_overall_rmse, rfmice_avg_overall_rmse]

            best_avg_missing_mae_idx = np.argmin(avg_missing_maes)
            best_avg_wasserstein_idx = np.argmin(avg_wasserstein_dists)
            best_avg_overall_mae_idx = np.argmin(avg_overall_maes)
            best_avg_overall_rmse_idx = np.argmin(avg_overall_rmses)

            # Print average results
            print("\nAverage Results Across All Runs:")
            print("Method          | Missing-only MAE | Wasserstein Dist | Overall MAE    | Overall RMSE")
            print("-" * 80)
            print(f"NeuralGradFlow  | {ngf_avg_missing_mae:.6f}±{ngf_std_missing_mae:.6f} | {ngf_avg_wass:.6f}       | {ngf_avg_overall_mae:.6f}   | {ngf_avg_overall_rmse:.6f}")
            print(f"MICE            | {mice_avg_missing_mae:.6f}±{mice_std_missing_mae:.6f} | {mice_avg_wass:.6f}       | {mice_avg_overall_mae:.6f}   | {mice_avg_overall_rmse:.6f}")
            print(f"RFMICE          | {rfmice_avg_missing_mae:.6f}±{rfmice_std_missing_mae:.6f} | {rfmice_avg_wass:.6f}       | {rfmice_avg_overall_mae:.6f}   | {rfmice_avg_overall_rmse:.6f}")
            print("-" * 80)

            print(f"\nBest method by Average Missing-only MAE: {methods[best_avg_missing_mae_idx]}")
            print(f"Best method by Average Wasserstein distance: {methods[best_avg_wasserstein_idx]}")
            print(f"Best method by Average Overall MAE: {methods[best_avg_overall_mae_idx]}")
            print(f"Best method by Average Overall RMSE: {methods[best_avg_overall_rmse_idx]}")
            print("-" * 40)

            # Print individual run results
            print("\nIndividual Run Results:")
            for run in range(NUM_RUNS):
                run_df = scenario_df[scenario_df["run"] == run + 1]
                if not run_df.empty:
                    print(f"\nRun {run+1}:")
                    print("Method          | Missing-only MAE | Wasserstein Dist | Overall MAE    | Overall RMSE")
                    print("-" * 80)
                    print(f"NeuralGradFlow  | {run_df['mae'].values[0]:.6f}      | {run_df['wass'].values[0]:.6f}       | {run_df['overall_mae'].values[0]:.6f}   | {run_df['overall_rmse'].values[0]:.6f}")
                    print(f"MICE            | {run_df['mice_missing_mae'].values[0]:.6f}      | {run_df['mice_wass'].values[0]:.6f}       | {run_df['mice_overall_mae'].values[0]:.6f}   | {run_df['mice_overall_rmse'].values[0]:.6f}")
                    print(f"RFMICE          | {run_df['rfmice_missing_mae'].values[0]:.6f}      | {run_df['rfmice_wass'].values[0]:.6f}       | {run_df['rfmice_overall_mae'].values[0]:.6f}   | {run_df['rfmice_overall_rmse'].values[0]:.6f}")
                    print("-" * 40)

# Create a DataFrame with average results
avg_results = []
for scenario in SCENARIO:
    for p_miss in P_MISS:
        avg_data = {
            "missing": scenario,
            "p_miss": p_miss,
            "dataset_name": args.dataset_name,
            "seed": args.seed,
            "run": "average",
            "interval": int(args.iter_time - 1),
            "mae": np.mean(all_runs_results['NeuralGradFlow']['missing_mae'][scenario][p_miss]),
            "wass": np.mean(all_runs_results['NeuralGradFlow']['wass'][scenario][p_miss]),
            "overall_mae": np.mean(all_runs_results['NeuralGradFlow']['overall_mae'][scenario][p_miss]),
            "overall_rmse": np.mean(all_runs_results['NeuralGradFlow']['overall_rmse'][scenario][p_miss]),
            "mice_missing_mae": np.mean(all_runs_results['MICE']['missing_mae'][scenario][p_miss]),
            "mice_wass": np.mean(all_runs_results['MICE']['wass'][scenario][p_miss]),
            "mice_overall_mae": np.mean(all_runs_results['MICE']['overall_mae'][scenario][p_miss]),
            "mice_overall_rmse": np.mean(all_runs_results['MICE']['overall_rmse'][scenario][p_miss]),
            "rfmice_missing_mae": np.mean(all_runs_results['RFMICE']['missing_mae'][scenario][p_miss]),
            "rfmice_wass": np.mean(all_runs_results['RFMICE']['wass'][scenario][p_miss]),
            "rfmice_overall_mae": np.mean(all_runs_results['RFMICE']['overall_mae'][scenario][p_miss]),
            "rfmice_overall_rmse": np.mean(all_runs_results['RFMICE']['overall_rmse'][scenario][p_miss]),
            # Add standard deviations
            "mae_std": np.std(all_runs_results['NeuralGradFlow']['missing_mae'][scenario][p_miss]),
            "mice_missing_mae_std": np.std(all_runs_results['MICE']['missing_mae'][scenario][p_miss]),
            "rfmice_missing_mae_std": np.std(all_runs_results['RFMICE']['missing_mae'][scenario][p_miss])
        }

        # Determine best methods based on average metrics
        methods = ["NeuralGradFlow", "MICE", "RFMICE"]
        avg_missing_maes = [
            np.mean(all_runs_results['NeuralGradFlow']['missing_mae'][scenario][p_miss]),
            np.mean(all_runs_results['MICE']['missing_mae'][scenario][p_miss]),
            np.mean(all_runs_results['RFMICE']['missing_mae'][scenario][p_miss])
        ]
        avg_wasserstein_dists = [
            np.mean(all_runs_results['NeuralGradFlow']['wass'][scenario][p_miss]),
            np.mean(all_runs_results['MICE']['wass'][scenario][p_miss]),
            np.mean(all_runs_results['RFMICE']['wass'][scenario][p_miss])
        ]
        avg_overall_maes = [
            np.mean(all_runs_results['NeuralGradFlow']['overall_mae'][scenario][p_miss]),
            np.mean(all_runs_results['MICE']['overall_mae'][scenario][p_miss]),
            np.mean(all_runs_results['RFMICE']['overall_mae'][scenario][p_miss])
        ]
        avg_overall_rmses = [
            np.mean(all_runs_results['NeuralGradFlow']['overall_rmse'][scenario][p_miss]),
            np.mean(all_runs_results['MICE']['overall_rmse'][scenario][p_miss]),
            np.mean(all_runs_results['RFMICE']['overall_rmse'][scenario][p_miss])
        ]

        avg_data["best_method_missing_mae"] = methods[np.argmin(avg_missing_maes)]
        avg_data["best_method_wasserstein"] = methods[np.argmin(avg_wasserstein_dists)]
        avg_data["best_method_overall_mae"] = methods[np.argmin(avg_overall_maes)]
        avg_data["best_method_overall_rmse"] = methods[np.argmin(avg_overall_rmses)]

        avg_results.append(avg_data)

# Add average results to the DataFrame
avg_df = pd.DataFrame(avg_results)
result_df = pd.concat([result_df, avg_df], axis=0)

print("\nComplete Results DataFrame:")
print(result_df)

# Save results to CSV
result_df.to_csv(os.path.join(args.outpath, csv_name), index=None)
print(f"\nResults saved to {os.path.join(args.outpath, csv_name)}")

# Also save a summary file with just the average results
avg_csv_name = f"average_{csv_name}"
avg_df.to_csv(os.path.join(args.outpath, avg_csv_name), index=None)
print(f"Average results saved to {os.path.join(args.outpath, avg_csv_name)}")

