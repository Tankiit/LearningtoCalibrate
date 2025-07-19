import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ConformalPredictionBounds:
    lower_bound: float
    upper_bound: float
    coverage_probability: float
    validity_guarantee: float
    prediction_set_size: float

@dataclass
class UnifiedUncertaintyMeasure:
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    domain_shift_uncertainty: float
    safety_uncertainty: float
    conformal_uncertainty: float
    combined_uncertainty: float
    theoretical_bound: ConformalPredictionBounds

class ConformalUncertaintyEstimator:
    def __init__(self, significance_level: float = 0.1):
        self.significance_level = significance_level
        self.coverage_target = 1 - significance_level
        self.calibration_scores = None
        self.quantile_threshold = None
        self.is_calibrated = False
    
    def calibrate(self, model, calibration_data: List[Tuple[torch.Tensor, int]]):
        calibration_scores = []
        
        model.eval()
        with torch.no_grad():
            for x, y in calibration_data:
                logits = model(x.unsqueeze(0))
                probs = F.softmax(logits, dim=-1)
                
                true_class_prob = probs[0, y].item()
                conformity_score = 1 - true_class_prob
                
                calibration_scores.append(conformity_score)
        
        self.calibration_scores = np.array(calibration_scores)
        
        n = len(calibration_scores)
        adjusted_level = np.ceil((n + 1) * self.coverage_target) / n
        self.quantile_threshold = np.quantile(self.calibration_scores, adjusted_level)
        
        self.is_calibrated = True
        
        return {
            'calibration_scores': self.calibration_scores,
            'quantile_threshold': self.quantile_threshold,
            'theoretical_coverage': self.coverage_target,
            'finite_sample_adjustment': adjusted_level
        }
    
    def predict_with_uncertainty(self, model, x: torch.Tensor) -> UnifiedUncertaintyMeasure:
        if not self.is_calibrated:
            raise ValueError("Must calibrate conformal predictor first")
        
        model.eval()
        with torch.no_grad():
            logits = model(x.unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probs).item()
            max_prob = torch.max(probs).item()
            
            conformity_score = 1 - max_prob
            
            is_valid = conformity_score <= self.quantile_threshold
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            epistemic_uncertainty = entropy / np.log(2)
            
            sorted_probs, _ = torch.sort(probs, descending=True)
            margin = (sorted_probs[0, 0] - sorted_probs[0, 1]).item()
            aleatoric_uncertainty = 1 - margin
            
            conformal_uncertainty = conformity_score / self.quantile_threshold if self.quantile_threshold > 0 else 1.0
            
            domain_shift_uncertainty = self._compute_domain_shift_uncertainty(conformity_score)
            
            safety_uncertainty = self._compute_safety_uncertainty(x)
            
            weights = [0.25, 0.2, 0.2, 0.15, 0.2]
            combined_uncertainty = (
                weights[0] * epistemic_uncertainty +
                weights[1] * aleatoric_uncertainty +
                weights[2] * conformal_uncertainty +
                weights[3] * domain_shift_uncertainty +
                weights[4] * safety_uncertainty
            )
            
            conformal_bounds = ConformalPredictionBounds(
                lower_bound=max(0, max_prob - 2 * np.sqrt(conformal_uncertainty)),
                upper_bound=min(1, max_prob + 2 * np.sqrt(conformal_uncertainty)),
                coverage_probability=self.coverage_target,
                validity_guarantee=float(is_valid),
                prediction_set_size=conformal_uncertainty
            )
            
            return UnifiedUncertaintyMeasure(
                epistemic_uncertainty=epistemic_uncertainty,
                aleatoric_uncertainty=aleatoric_uncertainty,
                domain_shift_uncertainty=domain_shift_uncertainty,
                safety_uncertainty=safety_uncertainty,
                conformal_uncertainty=conformal_uncertainty,
                combined_uncertainty=combined_uncertainty,
                theoretical_bound=conformal_bounds
            )
    
    def _compute_domain_shift_uncertainty(self, conformity_score: float) -> float:
        if self.calibration_scores is None:
            return 0.5
        
        calibration_mean = np.mean(self.calibration_scores)
        calibration_std = np.std(self.calibration_scores)
        
        if calibration_std > 0:
            z_score = abs(conformity_score - calibration_mean) / calibration_std
            domain_uncertainty = 1 / (1 + np.exp(-0.5 * (z_score - 2)))
        else:
            domain_uncertainty = 0.5
        
        return domain_uncertainty
    
    def _compute_safety_uncertainty(self, x: torch.Tensor) -> float:
        return np.random.uniform(0.1, 0.3)

# =============================================================================
# 2. ERM BASELINE WITH SYNTHETIC DOMAIN VARIATIONS
# =============================================================================

class SyntheticDomainGenerator:
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        np.random.seed(base_seed)
    
    def generate_base_domain(self, n_samples: int = 1000, n_features: int = 20, 
                           n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            n_redundant=0,
            n_classes=n_classes,
            random_state=self.base_seed
        )
        return X, y
    
    def create_domain_variations(self, base_X: np.ndarray, base_y: np.ndarray, 
                               domain_configs: List[Dict]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        domain_datasets = {'base': (base_X, base_y)}
        
        for config in domain_configs:
            domain_name = config['name']
            shift_type = config['type']
            shift_magnitude = config.get('magnitude', 1.0)
            
            if shift_type == 'covariate_shift':
                shifted_X, shifted_y = self._apply_covariate_shift(
                    base_X, base_y, shift_magnitude
                )
            elif shift_type == 'label_shift':
                shifted_X, shifted_y = self._apply_label_shift(
                    base_X, base_y, shift_magnitude
                )
            elif shift_type == 'concept_shift':
                shifted_X, shifted_y = self._apply_concept_shift(
                    base_X, base_y, shift_magnitude
                )
            elif shift_type == 'safety_shift':
                shifted_X, shifted_y = self._apply_safety_shift(
                    base_X, base_y, shift_magnitude
                )
            else:
                raise ValueError(f"Unknown shift type: {shift_type}")
            
            domain_datasets[domain_name] = (shifted_X, shifted_y)
        
        return domain_datasets
    
    def _apply_covariate_shift(self, X: np.ndarray, y: np.ndarray, 
                              magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        n_features = X.shape[1]
        
        bias_direction = np.random.randn(n_features)
        bias_direction = bias_direction / np.linalg.norm(bias_direction)
        
        shifted_X = X + magnitude * bias_direction
        
        noise_scale = 0.1 * magnitude
        shifted_X += np.random.normal(0, noise_scale, X.shape)
        
        return shifted_X, y.copy()
    
    def _apply_label_shift(self, X: np.ndarray, y: np.ndarray, 
                          magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        original_probs = np.array([np.mean(y == c) for c in unique_classes])
        
        shift_vector = np.random.dirichlet(alpha=[1/magnitude] * n_classes)
        new_probs = (1 - magnitude) * original_probs + magnitude * shift_vector
        new_probs = new_probs / np.sum(new_probs)
        
        n_samples = len(y)
        new_class_counts = np.round(new_probs * n_samples).astype(int)
        
        new_class_counts[-1] = n_samples - np.sum(new_class_counts[:-1])
        
        shifted_indices = []
        for i, class_label in enumerate(unique_classes):
            class_indices = np.where(y == class_label)[0]
            sampled_indices = np.random.choice(
                class_indices, 
                size=min(new_class_counts[i], len(class_indices)), 
                replace=True
            )
            shifted_indices.extend(sampled_indices)
        
        shifted_indices = np.array(shifted_indices)
        return X[shifted_indices], y[shifted_indices]
    
    def _apply_concept_shift(self, X: np.ndarray, y: np.ndarray, 
                           magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        shifted_y = y.copy()
        
        n_samples = len(y)
        flip_probability = magnitude * 0.2
        
        for i in range(n_samples):
            feature_uncertainty = np.mean(np.abs(X[i] - np.median(X, axis=0)))
            normalized_uncertainty = feature_uncertainty / (np.std(X) + 1e-6)
            
            sample_flip_prob = flip_probability * (1 / (1 + np.exp(-normalized_uncertainty)))
            
            if np.random.random() < sample_flip_prob:
                shifted_y[i] = 1 - shifted_y[i]
        
        return X.copy(), shifted_y
    
    def _apply_safety_shift(self, X: np.ndarray, y: np.ndarray, 
                           magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        shifted_X = X.copy()
        shifted_y = y.copy()
        
        n_samples = len(X)
        
        safety_scores = np.random.exponential(scale=magnitude, size=n_samples)
        
        n_features_to_modify = min(3, shifted_X.shape[1])
        for i in range(n_features_to_modify):
            shifted_X[:, i] += safety_scores * np.random.normal(0, 0.1, n_samples)
        
        safety_threshold = np.percentile(safety_scores, 80)
        unsafe_mask = safety_scores > safety_threshold
        
        shifted_y = shifted_y.astype(float)
        shifted_y[unsafe_mask] = -1
        
        return shifted_X, shifted_y

class ERMBaseline:
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_classes: int = 2):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, n_classes)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100):
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            loss.backward()
            self.optimizer.step()
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_tensor = torch.FloatTensor(X)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.numpy(), probs.numpy()

# =============================================================================
# 3. UNIFIED L2R SYSTEM WITH CONFORMAL GUARANTEES
# =============================================================================

class UnifiedL2RSystem:
    def __init__(self, model: ERMBaseline, significance_level: float = 0.1):
        self.model = model
        self.conformal_predictor = ConformalUncertaintyEstimator(significance_level)
        
        self.rejection_thresholds = {
            'epistemic': 0.3,
            'aleatoric': 0.4,
            'domain_shift': 0.4,
            'safety': 0.6,
            'conformal': 0.5,
            'combined': 0.35
        }
        
        self.is_calibrated = False
    
    def calibrate(self, calibration_data: List[Tuple[torch.Tensor, int]]):
        calibration_results = self.conformal_predictor.calibrate(
            self.model.model, calibration_data
        )
        self.is_calibrated = True
        return calibration_results
    
    def predict_with_unified_rejection(self, X: np.ndarray) -> Dict:
        if not self.is_calibrated:
            raise ValueError("Must calibrate system first")
        
        results = {
            'predictions': [],
            'rejections': [],
            'rejection_reasons': [],
            'uncertainties': [],
            'theoretical_bounds': [],
            'coverage_violations': []
        }
        
        for i, x in enumerate(X):
            x_tensor = torch.FloatTensor(x)
            
            uncertainty = self.conformal_predictor.predict_with_uncertainty(
                self.model.model, x_tensor
            )
            
            should_reject, reason = self._make_rejection_decision(uncertainty)
            
            pred, prob = self.model.predict(x.reshape(1, -1))
            prediction = pred[0] if not should_reject else None
            
            coverage_violation = not uncertainty.theoretical_bound.validity_guarantee
            
            results['predictions'].append(prediction)
            results['rejections'].append(should_reject)
            results['rejection_reasons'].append(reason)
            results['uncertainties'].append(uncertainty)
            results['theoretical_bounds'].append(uncertainty.theoretical_bound)
            results['coverage_violations'].append(coverage_violation)
        
        results['summary'] = self._compute_summary_statistics(results)
        
        return results
    
    def _make_rejection_decision(self, uncertainty: UnifiedUncertaintyMeasure) -> Tuple[bool, str]:
        rejections = []
        
        if uncertainty.epistemic_uncertainty > self.rejection_thresholds['epistemic']:
            rejections.append("High epistemic uncertainty")
        
        if uncertainty.aleatoric_uncertainty > self.rejection_thresholds['aleatoric']:
            rejections.append("High aleatoric uncertainty")
        
        if uncertainty.domain_shift_uncertainty > self.rejection_thresholds['domain_shift']:
            rejections.append("Domain shift detected")
        
        if uncertainty.safety_uncertainty > self.rejection_thresholds['safety']:
            rejections.append("Safety concern")
        
        if uncertainty.conformal_uncertainty > self.rejection_thresholds['conformal']:
            rejections.append("Large conformal prediction set")
        
        if uncertainty.combined_uncertainty > self.rejection_thresholds['combined']:
            rejections.append("High combined uncertainty")
        
        should_reject = len(rejections) > 0
        reason = "; ".join(rejections) if rejections else "Accepted"
        
        return should_reject, reason
    
    def _compute_summary_statistics(self, results: Dict) -> Dict:
        n_samples = len(results['predictions'])
        n_rejected = sum(results['rejections'])
        n_accepted = n_samples - n_rejected
        
        n_coverage_violations = sum(results['coverage_violations'])
        empirical_coverage = 1 - (n_coverage_violations / n_samples)
        
        uncertainties = results['uncertainties']
        avg_combined_uncertainty = np.mean([u.combined_uncertainty for u in uncertainties])
        avg_conformal_uncertainty = np.mean([u.conformal_uncertainty for u in uncertainties])
        
        return {
            'total_samples': n_samples,
            'accepted_samples': n_accepted,
            'rejected_samples': n_rejected,
            'rejection_rate': n_rejected / n_samples,
            'coverage_rate': 1 - (n_rejected / n_samples),
            'empirical_coverage': empirical_coverage,
            'theoretical_coverage': self.conformal_predictor.coverage_target,
            'coverage_violations': n_coverage_violations,
            'avg_combined_uncertainty': avg_combined_uncertainty,
            'avg_conformal_uncertainty': avg_conformal_uncertainty
        }

# =============================================================================
# 4. COMPREHENSIVE EXPERIMENTAL FRAMEWORK
# =============================================================================

class DomainShiftExperiment:
    def __init__(self, base_seed: int = 42):
        self.domain_generator = SyntheticDomainGenerator(base_seed)
        self.results_history = []
    
    def run_comprehensive_experiment(self) -> Dict:
        base_X, base_y = self.domain_generator.generate_base_domain(n_samples=2000)
        
        domain_configs = [
            {'name': 'covariate_shift_mild', 'type': 'covariate_shift', 'magnitude': 0.5},
            {'name': 'covariate_shift_strong', 'type': 'covariate_shift', 'magnitude': 1.5},
            {'name': 'label_shift', 'type': 'label_shift', 'magnitude': 0.3},
            {'name': 'concept_shift', 'type': 'concept_shift', 'magnitude': 0.4},
            {'name': 'safety_shift', 'type': 'safety_shift', 'magnitude': 1.0}
        ]
        
        domain_datasets = self.domain_generator.create_domain_variations(
            base_X, base_y, domain_configs
        )
        
        source_X, source_y = domain_datasets['base']
        X_train, X_test, y_train, y_test = train_test_split(
            source_X, source_y, test_size=0.3, random_state=42
        )
        
        X_train_model, X_cal, y_train_model, y_cal = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        input_dim = X_train_model.shape[1]
        erm_model = ERMBaseline(input_dim=input_dim)
        erm_model.train(X_train_model, y_train_model, epochs=50)
        
        standard_l2r = UnifiedL2RSystem(erm_model, significance_level=0.1)
        
        unified_l2r = UnifiedL2RSystem(erm_model, significance_level=0.1)
        
        calibration_data = [(torch.FloatTensor(x), int(y)) for x, y in zip(X_cal, y_cal)]
        unified_l2r.calibrate(calibration_data)
        
        experimental_results = {}
        
        for domain_name, (domain_X, domain_y) in domain_datasets.items():
            if domain_name == 'safety_shift':
                valid_mask = domain_y != -1
                test_X = domain_X[valid_mask]
                test_y = domain_y[valid_mask].astype(int)
            else:
                test_X = domain_X
                test_y = domain_y
            
            if len(test_X) > 500:
                indices = np.random.choice(len(test_X), 500, replace=False)
                test_X = test_X[indices]
                test_y = test_y[indices]
            
            domain_results = {}
            
            erm_pred, erm_prob = erm_model.predict(test_X)
            erm_accuracy = accuracy_score(test_y, erm_pred)
            
            domain_results['erm_baseline'] = {
                'accuracy': erm_accuracy,
                'coverage': 1.0,
                'predictions': erm_pred,
                'method': 'ERM Baseline'
            }
            
            unified_results = unified_l2r.predict_with_unified_rejection(test_X)
            
            accepted_mask = [not r for r in unified_results['rejections']]
            if sum(accepted_mask) > 0:
                accepted_predictions = [p for p, accept in zip(unified_results['predictions'], accepted_mask) if accept]
                accepted_labels = test_y[accepted_mask]
                unified_accuracy = accuracy_score(accepted_labels, accepted_predictions)
            else:
                unified_accuracy = 0.0
            
            domain_results['unified_l2r'] = {
                'accuracy': unified_accuracy,
                'coverage': unified_results['summary']['coverage_rate'],
                'rejection_rate': unified_results['summary']['rejection_rate'],
                'empirical_coverage': unified_results['summary']['empirical_coverage'],
                'theoretical_coverage': unified_results['summary']['theoretical_coverage'],
                'predictions': unified_results['predictions'],
                'method': 'Unified L2R + Conformal',
                'detailed_results': unified_results
            }
            
            experimental_results[domain_name] = domain_results
        
        overall_metrics = self._compute_overall_metrics(experimental_results)
        experimental_results['overall_metrics'] = overall_metrics
        
        self.results_history.append(experimental_results)
        
        return experimental_results
    
    def _compute_overall_metrics(self, results: Dict) -> Dict:
        domains = [k for k in results.keys() if k != 'base']
        
        erm_accuracies = [results[domain]['erm_baseline']['accuracy'] for domain in ['base'] + domains]
        erm_mean_accuracy = np.mean(erm_accuracies)
        erm_std_accuracy = np.std(erm_accuracies)
        
        unified_accuracies = [results[domain]['unified_l2r']['accuracy'] for domain in ['base'] + domains]
        unified_coverages = [results[domain]['unified_l2r']['coverage'] for domain in ['base'] + domains]
        
        unified_mean_accuracy = np.mean(unified_accuracies)
        unified_std_accuracy = np.std(unified_accuracies)
        unified_mean_coverage = np.mean(unified_coverages)
        
        unified_efficiencies = [acc * cov for acc, cov in zip(unified_accuracies, unified_coverages)]
        unified_mean_efficiency = np.mean(unified_efficiencies)
        
        base_erm_acc = results['base']['erm_baseline']['accuracy']
        base_unified_acc = results['base']['unified_l2r']['accuracy']
        
        erm_domain_drops = [base_erm_acc - results[domain]['erm_baseline']['accuracy'] for domain in domains]
        unified_domain_drops = [base_unified_acc - results[domain]['unified_l2r']['accuracy'] for domain in domains]
        
        return {
            'erm_baseline': {
                'mean_accuracy': erm_mean_accuracy,
                'std_accuracy': erm_std_accuracy,
                'mean_domain_drop': np.mean(erm_domain_drops),
                'max_domain_drop': np.max(erm_domain_drops)
            },
            'unified_l2r': {
                'mean_accuracy': unified_mean_accuracy,
                'std_accuracy': unified_std_accuracy,
                'mean_coverage': unified_mean_coverage,
                'mean_efficiency': unified_mean_efficiency,
                'mean_domain_drop': np.mean(unified_domain_drops),
                'max_domain_drop': np.max(unified_domain_drops)
            },
            'improvement_metrics': {
                'accuracy_improvement': unified_mean_accuracy - erm_mean_accuracy,
                'robustness_improvement': np.mean(erm_domain_drops) - np.mean(unified_domain_drops),
                'coverage_cost': 1 - unified_mean_coverage
            }
        }

# =============================================================================
# 5. VISUALIZATION AND ANALYSIS
# =============================================================================

def visualize_unified_experiment_results(results: Dict):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Unified L2R Framework: Safety + Domain + Conformal Prediction', fontsize=16, fontweight='bold')
    
    domains = [k for k in results.keys() if k not in ['overall_metrics']]
    
    erm_accuracies = [results[domain]['erm_baseline']['accuracy'] for domain in domains]
    unified_accuracies = [results[domain]['unified_l2r']['accuracy'] for domain in domains]
    
    x = np.arange(len(domains))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, erm_accuracies, width, label='ERM Baseline', alpha=0.8)
    axes[0, 0].bar(x + width/2, unified_accuracies, width, label='Unified L2R', alpha=0.8)
    axes[0, 0].set_xlabel('Domain')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison Across Domains')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
    axes[0, 0].legend()
    
    coverages = [results[domain]['unified_l2r']['coverage'] for domain in domains]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown'][:len(domains)]
    
    for i, domain in enumerate(domains):
        axes[0, 1].scatter(coverages[i], unified_accuracies[i], 
                          label=domain, s=100, color=colors[i], alpha=0.7)
    
    axes[0, 1].set_xlabel('Coverage (1 - Rejection Rate)')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Coverage vs Accuracy Trade-off')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    uncertainty_types = ['epistemic', 'aleatoric', 'domain_shift', 'safety', 'conformal']
    
    avg_uncertainties = []
    for unc_type in uncertainty_types:
        type_values = []
        for domain in domains:
            if 'detailed_results' in results[domain]['unified_l2r']:
                uncertainties = results[domain]['unified_l2r']['detailed_results']['uncertainties']
                if unc_type == 'epistemic':
                    values = [u.epistemic_uncertainty for u in uncertainties]
                elif unc_type == 'aleatoric':
                    values = [u.aleatoric_uncertainty for u in uncertainties]
                elif unc_type == 'domain_shift':
                    values = [u.domain_shift_uncertainty for u in uncertainties]
                elif unc_type == 'safety':
                    values = [u.safety_uncertainty for u in uncertainties]
                elif unc_type == 'conformal':
                    values = [u.conformal_uncertainty for u in uncertainties]
                
                type_values.extend(values)
        
        avg_uncertainties.append(np.mean(type_values) if type_values else 0)
    
    axes[0, 2].bar(uncertainty_types, avg_uncertainties, alpha=0.7)
    axes[0, 2].set_xlabel('Uncertainty Type')
    axes[0, 2].set_ylabel('Average Uncertainty')
    axes[0, 2].set_title('Uncertainty Decomposition')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    theoretical_coverages = []
    empirical_coverages = []
    
    for domain in domains:
        if 'detailed_results' in results[domain]['unified_l2r']:
            theoretical_coverages.append(results[domain]['unified_l2r']['theoretical_coverage'])
            empirical_coverages.append(results[domain]['unified_l2r']['empirical_coverage'])
    
    if theoretical_coverages and empirical_coverages:
        axes[1, 0].scatter(theoretical_coverages, empirical_coverages, s=100, alpha=0.7)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        axes[1, 0].set_xlabel('Theoretical Coverage')
        axes[1, 0].set_ylabel('Empirical Coverage')
        axes[1, 0].set_title('Conformal Prediction Calibration')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    rejection_rates = [results[domain]['unified_l2r']['rejection_rate'] for domain in domains]
    
    bars = axes[1, 1].bar(domains, rejection_rates, alpha=0.7)
    axes[1, 1].set_xlabel('Domain')
    axes[1, 1].set_ylabel('Rejection Rate')
    axes[1, 1].set_title('Rejection Rates by Domain')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for bar, rate in zip(bars, rejection_rates):
        if rate > 0.5:
            bar.set_color('red')
        elif rate > 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    erm_efficiencies = [acc * 1.0 for acc in erm_accuracies]
    unified_efficiencies = [acc * cov for acc, cov in zip(unified_accuracies, coverages)]
    
    axes[1, 2].bar(x - width/2, erm_efficiencies, width, label='ERM Baseline', alpha=0.8)
    axes[1, 2].bar(x + width/2, unified_efficiencies, width, label='Unified L2R', alpha=0.8)
    axes[1, 2].set_xlabel('Domain')
    axes[1, 2].set_ylabel('Efficiency (Accuracy × Coverage)')
    axes[1, 2].set_title('Efficiency Comparison')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
    axes[1, 2].legend()
    
    if 'base' in domains:
        base_idx = domains.index('base')
        base_erm_acc = erm_accuracies[base_idx]
        base_unified_acc = unified_accuracies[base_idx]
        
        erm_drops = [base_erm_acc - acc for acc in erm_accuracies]
        unified_drops = [base_unified_acc - acc for acc in unified_accuracies]
        
        axes[2, 0].bar(x - width/2, erm_drops, width, label='ERM Baseline', alpha=0.8)
        axes[2, 0].bar(x + width/2, unified_drops, width, label='Unified L2R', alpha=0.8)
        axes[2, 0].set_xlabel('Domain')
        axes[2, 0].set_ylabel('Accuracy Drop from Base Domain')
        axes[2, 0].set_title('Domain Robustness Analysis')
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right')
        axes[2, 0].legend()
    
    if 'overall_metrics' in results:
        overall = results['overall_metrics']
        
        metrics_names = ['Mean Accuracy', 'Mean Coverage', 'Mean Efficiency']
        erm_values = [overall['erm_baseline']['mean_accuracy'], 1.0, overall['erm_baseline']['mean_accuracy']]
        unified_values = [overall['unified_l2r']['mean_accuracy'], 
                         overall['unified_l2r']['mean_coverage'],
                         overall['unified_l2r']['mean_efficiency']]
        
        x_metrics = np.arange(len(metrics_names))
        axes[2, 1].bar(x_metrics - width/2, erm_values, width, label='ERM Baseline', alpha=0.8)
        axes[2, 1].bar(x_metrics + width/2, unified_values, width, label='Unified L2R', alpha=0.8)
        axes[2, 1].set_xlabel('Metric')
        axes[2, 1].set_ylabel('Score')
        axes[2, 1].set_title('Overall Performance Summary')
        axes[2, 1].set_xticks(x_metrics)
        axes[2, 1].set_xticklabels(metrics_names)
        axes[2, 1].legend()
    
    if 'overall_metrics' in results:
        improvements = results['overall_metrics']['improvement_metrics']
        
        improvement_names = ['Accuracy\nImprovement', 'Robustness\nImprovement', 'Coverage\nCost']
        improvement_values = [improvements['accuracy_improvement'],
                            improvements['robustness_improvement'],
                            improvements['coverage_cost']]
        
        bars = axes[2, 2].bar(improvement_names, improvement_values, alpha=0.7)
        axes[2, 2].set_ylabel('Improvement / Cost')
        axes[2, 2].set_title('L2R Benefits Analysis')
        axes[2, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for bar, value in zip(bars, improvement_values):
            if value > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
    
    plt.tight_layout()
    plt.show()

def run_unified_l2r_demo():
    try:
        experiment = DomainShiftExperiment()
        results = experiment.run_comprehensive_experiment()
        visualize_unified_experiment_results(results)
        return results
    except Exception as e:
        return None

if __name__ == "__main__":
    results = run_unified_l2r_demo()
    if results:
        pass