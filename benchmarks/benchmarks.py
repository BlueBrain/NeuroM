import os

import neurom as nm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '../test_data/')


class TimeLoadMorphology:
    def time_swc(self):
        path = os.path.join(DATA_DIR, 'swc/Neuron.swc')
        nm.load_neuron(path)

    def time_neurolucida_asc(self):
        path = os.path.join(DATA_DIR, 'neurolucida/bio_neuron-000.asc')
        nm.load_neuron(path)

    def time_h5(self):
        path = os.path.join(DATA_DIR, 'h5/v1/bio_neuron-000.h5')
        nm.load_neuron(path)


class TimeFeatures:
    def setup(self):
        path = os.path.join(DATA_DIR, 'h5/v1/bio_neuron-000.h5')
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
