"""Facade kompatibilitas untuk modul analisis yang telah dipisahkan."""

from .base import _family_legend as _family_legend
from .base import _save_figure as _save_figure
from .base import configure_plots as configure_plots
from .distributions import plot_class_distribution as plot_class_distribution
from .distributions import plot_feature_effect_sizes as plot_feature_effect_sizes
from .distributions import plot_non_ascii_characters as plot_non_ascii_characters
from .distributions import plot_text_length_by_model as plot_text_length_by_model
from .errors import plot_class_recall as plot_class_recall
from .errors import plot_confusion_matrix as plot_confusion_matrix
from .errors import plot_family_confusion as plot_family_confusion
from .style import plot_feature_correlation as plot_feature_correlation
from .style import plot_prompt_variability as plot_prompt_variability
from .style import plot_punctuation_heatmap as plot_punctuation_heatmap
from .style import plot_stylometry_heatmap as plot_stylometry_heatmap
from .style import plot_variance_explained as plot_variance_explained
