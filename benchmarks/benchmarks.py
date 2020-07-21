from pathlib import Path

import neurom as nm
import neurom.io
import neurom.fst._core
from neurom.check import neuron_checks as nc
from neurom.check import structural_checks as sc

DATA_DIR = Path(__file__).parent.parent / 'test_data/'


class TimeLoadMorphology(object):
    def time_swc(self):
        path = Path(DATA_DIR, 'swc/Neuron.swc')
        nm.load_neuron(path)

    def time_neurolucida_asc(self):
        path = Path(DATA_DIR, 'neurolucida/bio_neuron-000.asc')
        nm.load_neuron(path)

    def time_h5(self):
        path = Path(DATA_DIR, 'h5/v1/bio_neuron-000.h5')
        nm.load_neuron(path)


class TimeFeatures(object):
    def setup(self):
        path = Path(DATA_DIR, 'h5/v1/bio_neuron-000.h5')
        self.neuron = nm.load_neuron(path)

    def time_total_length(self):
        nm.get('total_length', self.neuron)

    def time_total_length_per_neurite(self):
        nm.get('total_length_per_neurite', self.neuron)

    def time_section_lengths(self):
        nm.get('section_lengths', self.neuron)

    def time_section_volumes(self):
        nm.get('section_volumes', self.neuron)

    def time_section_areas(self):
        nm.get('section_areas', self.neuron)

    def time_section_tortuosity(self):
        nm.get('section_tortuosity', self.neuron)

    def time_section_path_distances(self):
        nm.get('section_path_distances', self.neuron)

    def time_number_of_sections(self):
        nm.get('number_of_sections', self.neuron)

    def time_number_of_sections_per_neurite(self):
        nm.get('number_of_sections_per_neurite', self.neuron)

    def time_number_of_neurites(self):
        nm.get('number_of_neurites', self.neuron)

    def time_number_of_bifurcations(self):
        nm.get('number_of_bifurcations', self.neuron)

    def time_number_of_forking_points(self):
        nm.get('number_of_forking_points', self.neuron)

    def time_number_of_terminations(self):
        nm.get('number_of_terminations', self.neuron)

    def time_section_branch_orders(self):
        nm.get('section_branch_orders', self.neuron)

    def time_section_radial_distances(self):
        nm.get('section_radial_distances', self.neuron)

    def time_local_bifurcation_angles(self):
        nm.get('local_bifurcation_angles', self.neuron)

    def time_remote_bifurcation_angles(self):
        nm.get('remote_bifurcation_angles', self.neuron)

    def time_partition(self):
        nm.get('partition', self.neuron)

    def time_number_of_segments(self):
        nm.get('number_of_segments', self.neuron)

    def time_segment_lengths(self):
        nm.get('segment_lengths', self.neuron)

    def time_segment_radii(self):
        nm.get('segment_radii', self.neuron)

    def time_segment_midpoints(self):
        nm.get('segment_midpoints', self.neuron)

    def time_segment_taper_rates(self):
        nm.get('segment_taper_rates', self.neuron)

    def time_segment_radial_distances(self):
        nm.get('segment_radial_distances', self.neuron)

    def time_segment_meander_angles(self):
        nm.get('segment_meander_angles', self.neuron)

    def time_principal_direction_extents(self):
        nm.get('principal_direction_extents', self.neuron)

    def time_sholl_frequency(self):
        nm.get('sholl_frequency', self.neuron)


class TimeChecks:
    def setup(self):
        path = Path(DATA_DIR, 'h5/v1/bio_neuron-000.h5')
        self.data_wrapper = neurom.io.load_data(path)
        self.neuron = neurom.fst._core.FstNeuron(self.data_wrapper)

    def time_has_sequential_ids(self):
        sc.has_sequential_ids(self.data_wrapper)

    def time_no_missing_parents(self):
        sc.no_missing_parents(self.data_wrapper)

    def time_is_single_tree(self):
        sc.is_single_tree(self.data_wrapper)

    def time_has_increasing_ids(self):
        sc.has_increasing_ids(self.data_wrapper)

    def time_has_soma_points(self):
        sc.has_soma_points(self.data_wrapper)

    def time_has_all_finite_radius_neurites(self):
        sc.has_all_finite_radius_neurites(self.data_wrapper, threshold=0.0)

    def time_has_valid_soma(self):
        sc.has_valid_soma(self.data_wrapper)

    def time_has_valid_neurites(self):
        sc.has_valid_neurites(self.data_wrapper)

    def time_has_axon(self):
        nc.has_axon(self.neuron)

    def time_has_apical_dendrite(self):
        nc.has_apical_dendrite(self.neuron, min_number=1)

    def time_has_basal_dendrite(self):
        nc.has_basal_dendrite(self.neuron, min_number=1)

    def time_has_no_flat_neurites(self):
        nc.has_no_flat_neurites(self.neuron, tol=0.1, method='ratio')

    def time_has_all_monotonic_neurites(self):
        nc.has_all_monotonic_neurites(self.neuron, tol=1e-6)

    def time_has_all_nonzero_segment_lengths(self):
        nc.has_all_nonzero_segment_lengths(self.neuron, threshold=0.0)

    def time_has_all_nonzero_section_lengths(self):
        nc.has_all_nonzero_section_lengths(self.neuron, threshold=0.0)

    def time_has_all_nonzero_neurite_radii(self):
        nc.has_all_nonzero_neurite_radii(self.neuron, threshold=0.0)

    def time_has_nonzero_soma_radius(self):
        nc.has_nonzero_soma_radius(self.neuron, threshold=0.0)

    def time_has_no_jumps(self):
        nc.has_no_jumps(self.neuron, max_distance=30.0, axis='z')

    def time_has_no_fat_ends(self):
        nc.has_no_fat_ends(self.neuron, multiple_of_mean=2.0, final_point_count=5)
