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

# setup random seed
enable_reproducible_results(args.seed)
print(f"we are running at: {args.seed}")
X = ground_truth
imputation_scenarios = simulate_scenarios(X, mechanisms=SCENARIO, percentages=P_MISS)

result_df = pd.DataFrame()
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

        enable_reproducible_results(args.seed)
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

        print(f"NeuralGradFlowImputer - Overall MAE: {ngf_overall_mae:.6f}")
        print(f"NeuralGradFlowImputer - Overall RMSE: {ngf_overall_rmse:.6f}")
        print(f"NeuralGradFlowImputer - Missing-only MAE: {ngf_missing_mae:.6f}")
        print(f"NeuralGradFlowImputer - Wasserstein distance: {ngf_wass:.6f}")
        print("-" * 60)

        # 2. MICE (Multiple Imputation by Chained Equations)
        print(f"\n===== Running MICE for {scenario} with {p_miss*100}% missing data =====")
        enable_reproducible_results(args.seed)
        mice_imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=100,
            random_state=args.seed,
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

        print(f"MICE - Overall MAE: {mice_overall_mae:.6f}")
        print(f"MICE - Overall RMSE: {mice_overall_rmse:.6f}")
        print(f"MICE - Missing-only MAE: {mice_missing_mae:.6f}")
        print(f"MICE - Wasserstein distance: {mice_wass:.6f}")
        print("-" * 60)

        # 3. RFMICE (Random Forest MICE)
        print(f"\n===== Running RFMICE for {scenario} with {p_miss*100}% missing data =====")
        enable_reproducible_results(args.seed)
        rfmice_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=args.seed),
            max_iter=10,
            random_state=args.seed,
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

        # print(result_list)
        result_list = [pd.DataFrame(dict_idx, index=[0]) for dict_idx in result_list]
        temp_result_df = pd.concat(result_list, axis=0)
        temp_result_df["missing"] = scenario
        temp_result_df["dataset_name"] = args.dataset_name
        temp_result_df["seed"] = args.seed
        temp_result_df["p_miss"] = p_miss
        result_df = pd.concat([result_df, temp_result_df], axis=0)
        del x, x_miss, mask, model
        gc.collect()
        torch.cuda.empty_cache()


# print(result_df)
        # _, result_list = NeuralGradFlowImputer.fit_transform(x)
result_df = result_df[result_df["interval"] == int(args.iter_time - 1)].reset_index(drop=True)
os.makedirs(args.outpath) if not os.path.exists(args.outpath) else None
csv_name = (f"model_{args.model}_data_{args.dataset_name}_seed_{args.seed}_"
            f"_ode_step_{args.lr}_bandwidth_{args.bandwidth}_reg_{args.entropy_reg}"
            f"_score_{eval(args.mlp_hidden)[0]}_epoch_{args.score_net_epoch}.csv")
# Print a summary of the results with both missing-only and overall metrics
print("\n===== SUMMARY OF RESULTS =====")
for scenario in SCENARIO:
    for p_miss in P_MISS:
        scenario_df = result_df[(result_df["missing"] == scenario) & (result_df["p_miss"] == p_miss) & (result_df["interval"] == int(args.iter_time - 1))]
        if not scenario_df.empty:
            print(f"\nScenario: {scenario}, Missing: {p_miss*100}%")
            print("\nMethod Comparison:")
            print("Method          | Missing-only MAE | Wasserstein Dist | Overall MAE    | Overall RMSE")
            print("-" * 80)
            print(f"NeuralGradFlow  | {scenario_df['mae'].values[0]:.6f}      | {scenario_df['wass'].values[0]:.6f}       | {scenario_df['overall_mae'].values[0]:.6f}   | {scenario_df['overall_rmse'].values[0]:.6f}")
            print(f"MICE            | {scenario_df['mice_missing_mae'].values[0]:.6f}      | {scenario_df['mice_wass'].values[0]:.6f}       | {scenario_df['mice_overall_mae'].values[0]:.6f}   | {scenario_df['mice_overall_rmse'].values[0]:.6f}")
            print(f"RFMICE          | {scenario_df['rfmice_missing_mae'].values[0]:.6f}      | {scenario_df['rfmice_wass'].values[0]:.6f}       | {scenario_df['rfmice_overall_mae'].values[0]:.6f}   | {scenario_df['rfmice_overall_rmse'].values[0]:.6f}")
            print("-" * 80)

            print(f"\nBest method by Missing-only MAE: {scenario_df['best_method_missing_mae'].values[0]}")
            print(f"Best method by Wasserstein distance: {scenario_df['best_method_wasserstein'].values[0]}")
            print(f"Best method by Overall MAE: {scenario_df['best_method_overall_mae'].values[0]}")
            print(f"Best method by Overall RMSE: {scenario_df['best_method_overall_rmse'].values[0]}")
            print("-" * 40)

print("\nComplete Results DataFrame:")
print(result_df)
result_df.to_csv(os.path.join(args.outpath, csv_name), index=None)
print(f"\nResults saved to {os.path.join(args.outpath, csv_name)}")

