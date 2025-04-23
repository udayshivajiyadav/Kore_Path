import streamlit as st
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Material settings
material_settings = {
    'PLA': {'extruder_temp': (190, 220), 'bed_temp': (50, 60), 'print_speed': (40, 60)},
    'ABS': {'extruder_temp': (220, 250), 'bed_temp': (80, 100), 'print_speed': (40, 60)},
    'PETG': {'extruder_temp': (230, 250), 'bed_temp': (70, 90), 'print_speed': (30, 50)},
    'TPU': {'extruder_temp': (220, 240), 'bed_temp': (40, 60), 'print_speed': (20, 40)},
}

# GcodeReader class with improved layer and segment detection
class GcodeType:
    FDM_REGULAR = 1
    FDM_STRATASYS = 2
    LPBF_REGULAR = 3
    LPBF_SCODE = 4

class LayerError(Exception):
    pass

class GcodeReader:
    def __init__(self, filename=None, gcode_lines=None, filetype=GcodeType.FDM_REGULAR):
        """Initialize with either a filename or a list of G-code lines."""
        if filename is not None:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"{filename} does not exist!")
            with open(filename, 'r') as infile:
                self.gcode_lines = [line.strip() for line in infile.readlines()]
        elif gcode_lines is not None:
            self.gcode_lines = gcode_lines
        else:
            raise ValueError("Either filename or gcode_lines must be provided.")
        self.filetype = filetype
        self.n_segs = 0
        self.segs = []  # List of (x0, y0, x1, y1, z) tuples
        self.n_layers = 0
        self.seg_index_bars = []
        self.subpaths = None
        self.subpath_index_bars = []
        self.layer_commands = defaultdict(list)
        self._read()

    def _read(self):
        if self.filetype == GcodeType.FDM_REGULAR:
            self._read_fdm_regular()

    def _read_fdm_regular(self):
        """Parse FDM regular G-code with improved layer and segment detection."""
        temp = -float('inf')
        gxyzef = [temp, temp, temp, temp, temp, temp]  # G, X, Y, Z, E, F
        current_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Track current X, Y, Z, E, F
        d = dict(zip(['G', 'X', 'Y', 'Z', 'E', 'F'], range(6)))
        seg_count = 0
        current_layer = -1  # Start at -1 so first ;LAYER_CHANGE sets it to 0
        layer_change_count = 0  # Debug: Count ;LAYER_CHANGE comments

        for line in self.gcode_lines:
            if not line or line.startswith(';'):
                if ';LAYER_CHANGE' in line:
                    layer_change_count += 1
                    current_layer += 1
                    self.n_layers = current_layer + 1  # n_layers is 1-based
                    self.seg_index_bars.append(seg_count)
                    print(f"Debug: Layer change detected at line: {line}, current_layer = {current_layer}, n_layers = {self.n_layers}")
                self.layer_commands[current_layer].append(line)
                continue

            tokens = line.split()
            if not tokens or tokens[0] not in ['G0', 'G1', 'G92']:
                self.layer_commands[current_layer].append(line)
                continue

            old_gxyzef = gxyzef[:]
            old_pos = current_pos[:]

            # Update gxyzef with new values
            for token in tokens[1:]:
                if token[0] in d:
                    try:
                        gxyzef[d[token[0]]] = float(token[1:])
                    except ValueError:
                        continue

            # Update current position
            for i in range(len(current_pos)):
                current_pos[i] = gxyzef[i] if gxyzef[i] != temp else current_pos[i]

            # Handle G92 (e.g., E reset)
            if tokens[0] == 'G92':
                if 'E' in tokens[1]:
                    current_pos[4] = float(tokens[1][1:])
                    gxyzef[4] = current_pos[4]
                continue

            if tokens[0] in ['G0', 'G1']:
                # Detect segments: Any G1 move with X/Y change is considered
                has_xy_move = (current_pos[1] != old_pos[1] or current_pos[2] != old_pos[2])
                is_g1 = (tokens[0] == 'G1')
                if is_g1 and has_xy_move:
                    x0, y0 = old_pos[1:3]
                    x1, y1 = current_pos[1:3]
                    z = current_pos[3]
                    self.segs.append((x0, y0, x1, y1, z))
                    seg_count += 1
                self.layer_commands[current_layer].append(line)

        self.n_segs = len(self.segs)
        self.segs = np.array(self.segs) if self.segs else np.array([])
        self.seg_index_bars.append(self.n_segs)

        # Debug information
        print(f"Debug: Total ;LAYER_CHANGE comments = {layer_change_count}")
        print(f"Debug: Final n_layers = {self.n_layers}")
        print(f"Debug: n_segs = {self.n_segs}")
        print(f"Debug: seg_index_bars = {self.seg_index_bars}")
        print(f"Debug: First few segments = {self.segs[:5] if self.n_segs > 0 else 'None'}")

    def _compute_subpaths(self, xy_tolerance=0.001, z_tolerance=0.001):
        """Compute subpaths with tolerance for small gaps in coordinates."""
        self.subpaths = []
        self.subpath_index_bars = [0]
        if self.n_segs == 0:
            print("Debug: No segments to compute subpaths")
            return

        x0, y0, x1, y1, z = self.segs[0, :]
        xs, ys, zs = [x0, x1], [y0, y1], [z, z]
        current_layer = 0
        for i, (x0, y0, x1, y1, z) in enumerate(self.segs[1:, :], 1):
            while current_layer < len(self.seg_index_bars) - 1 and i >= self.seg_index_bars[current_layer + 1]:
                current_layer += 1
                self.subpath_index_bars.append(len(self.subpaths))
            # Use tolerance to check continuity
            if (abs(x0 - xs[-1]) > xy_tolerance or 
                abs(y0 - ys[-1]) > xy_tolerance or 
                abs(z - zs[-1]) > z_tolerance):
                self.subpaths.append((xs, ys, zs))
                xs, ys, zs = [x0, x1], [y0, y1], [z, z]
            else:
                xs.append(x1)
                ys.append(y1)
                zs.append(z)
        if len(xs) != 0:
            self.subpaths.append((xs, ys, zs))
        while len(self.subpath_index_bars) < len(self.seg_index_bars):
            self.subpath_index_bars.append(len(self.subpaths))

        print(f"Debug: Number of subpaths = {len(self.subpaths)}")
        print(f"Debug: subpath_index_bars = {self.subpath_index_bars}")
        print(f"Debug: First subpath = {self.subpaths[0] if self.subpaths else 'None'}")

    def plot_layer(self, layer=1):
        """Plot a single layer, falling back to raw segments if subpaths are empty."""
        if layer < 1 or layer > self.n_layers:
            raise LayerError(f"Layer number {layer} is invalid! (n_layers = {self.n_layers})")
        self._compute_subpaths()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Try plotting subpaths first
        if self.subpaths and len(self.subpath_index_bars) > layer:
            left, right = (self.subpath_index_bars[layer - 1], self.subpath_index_bars[layer])
            print(f"Debug: Plotting layer {layer}, subpath range [{left}, {right})")
            if left < right:
                for xs, ys, _ in self.subpaths[left:right]:
                    ax.plot(xs, ys, 'b-')
            else:
                print(f"Debug: No subpaths for layer {layer}, falling back to segments")
                # Fallback to plotting raw segments
                seg_left, seg_right = (self.seg_index_bars[layer - 1], self.seg_index_bars[layer])
                if seg_left < seg_right:
                    for x0, y0, x1, y1, _ in self.segs[seg_left:seg_right]:
                        ax.plot([x0, x1], [y0, y1], 'b-')
                else:
                    print(f"Debug: No segments for layer {layer}")
        else:
            print(f"Debug: No subpaths to plot for layer {layer}, falling back to segments")
            # Fallback to plotting raw segments
            seg_left, seg_right = (self.seg_index_bars[layer - 1], self.seg_index_bars[layer])
            if seg_left < seg_right:
                for x0, y0, x1, y1, _ in self.segs[seg_left:seg_right]:
                    ax.plot([x0, x1], [y0, y1], 'b-')
            else:
                print(f"Debug: No segments for layer {layer}")

        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    def plot_layer_path(self, layer=1, z_tolerance=0.05):
        """Plot the full path (travel and extrusion) for a specific layer using precomputed segments."""
        if layer < 1 or layer > self.n_layers:
            raise LayerError(f"Layer number {layer} is invalid! (n_layers = {self.n_layers})")
        
        self._compute_subpaths()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use precomputed segments instead of re-parsing G-code
        seg_left, seg_right = (self.seg_index_bars[layer - 1], self.seg_index_bars[layer])
        if seg_left >= seg_right:
            print(f"Debug: No segments for layer {layer}")
            ax.set_aspect('equal')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            return fig
        
        target_z = self.segs[seg_left][4] if seg_left < self.n_segs else None
        if target_z is None:
            print(f"Debug: No segments for layer {layer}, cannot determine target Z")
            ax.set_aspect('equal')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            return fig
        print(f"Debug: Target Z for layer {layer} = {target_z}")

        # Plot segments directly
        for x0, y0, x1, y1, z in self.segs[seg_left:seg_right]:
            if abs(z - target_z) <= z_tolerance:
                ax.plot([x0, x1], [y0, y1], 'b-', linewidth=2, 
                        label='Extrusion' if 'Extrusion' not in ax.get_legend_handles_labels()[1] else '')

        ax.set_aspect('equal')
        ax.legend()
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        return fig

    def optimize_paths(self):
        """Optimize subpath order within each layer to minimize travel distance."""
        if not self.subpaths:
            self._compute_subpaths()
        if not self.subpaths:
            print("Debug: No subpaths to optimize")
            return []
        optimized_commands = []
        current_pos = [0.0, 0.0, 0.0]
        travel_speed = 9000
        current_e = 0.0

        # Add initial setup commands
        for cmd in self.layer_commands[-1]:
            optimized_commands.append(cmd)

        for layer in range(self.n_layers):
            optimized_commands.append(f";LAYER_CHANGE")
            if self.seg_index_bars[layer] < self.n_segs:
                z = self.segs[self.seg_index_bars[layer]][4]
                optimized_commands.append(f"G1 Z{z:.3f} F5000")
            else:
                print(f"Debug: No segments for layer {layer}, skipping Z command")
                continue
            
            # Add non-movement commands for this layer
            for cmd in self.layer_commands[layer]:
                if not (cmd.startswith('G0') or cmd.startswith('G1')):
                    optimized_commands.append(cmd)
            
            left, right = (self.subpath_index_bars[layer], self.subpath_index_bars[layer + 1])
            if left == right:
                print(f"Debug: No subpaths for layer {layer}, skipping optimization")
                continue
            
            subpaths = self.subpaths[left:right]
            if not subpaths:
                print(f"Debug: Empty subpaths for layer {layer}, skipping optimization")
                continue
            
            unvisited = list(range(len(subpaths)))
            current_idx = 0
            optimized_order = [current_idx]
            unvisited.remove(current_idx)

            while unvisited:
                min_dist = float('inf')
                next_idx = None
                curr_end = subpaths[current_idx][0][-1], subpaths[current_idx][1][-1]
                for idx in unvisited:
                    next_start = subpaths[idx][0][0], subpaths[idx][1][0]
                    dist = np.hypot(curr_end[0] - next_start[0], curr_end[1] - next_start[1])
                    if dist < min_dist:
                        min_dist = dist
                        next_idx = idx
                current_idx = next_idx
                optimized_order.append(current_idx)
                unvisited.remove(current_idx)

            for idx in optimized_order:
                xs, ys, zs = subpaths[idx]
                if (current_pos[0] != xs[0] or current_pos[1] != ys[0] or current_pos[2] != zs[0]):
                    optimized_commands.append(f"G0 X{xs[0]:.3f} Y{ys[0]:.3f} Z{zs[0]:.3f} F{travel_speed}")
                for i in range(1, len(xs)):
                    current_e += 0.1
                    optimized_commands.append(f"G1 X{xs[i]:.3f} Y{ys[i]:.3f} Z{zs[i]:.3f} E{current_e:.5f}")
                current_pos = [xs[-1], ys[-1], zs[-1]]

        # Add final commands
        for cmd in self.layer_commands[self.n_layers]:
            optimized_commands.append(cmd)
        
        print(f"Debug: First few optimized commands = {optimized_commands[:10]}")
        return optimized_commands

# Parse G-code settings
def parse_gcode(gcode_str):
    lines = gcode_str.split('\n')
    settings = {}
    for line in lines:
        if line.startswith(';'):
            continue
        match = re.match(r'([GM]\d+)\s*(.*)', line)
        if match:
            command = match.group(1)
            params_str = match.group(2)
            params = {}
            for param in re.findall(r'([XYZEFST])\s*(-?\d+\.?\d*)', params_str):
                params[param[0]] = float(param[1])
            if command in ['M104', 'M109'] and 'S' in params and 'extruder_temp' not in settings:
                settings['extruder_temp'] = params['S']
            elif command in ['M140', 'M190'] and 'S' in params and 'bed_temp' not in settings:
                settings['bed_temp'] = params['S']
            elif command == 'G0' and 'F' in params and 'travel_speed' not in settings:
                settings['travel_speed'] = params['F'] / 60
    return settings

# Apply optimizations
def apply_optimizations(gcode_lines, selected_opts, reader):
    optimized_lines = gcode_lines.copy()
    if "Optimize travel paths" in selected_opts:
        optimized_lines = reader.optimize_paths()
        selected_opts.remove("Optimize travel paths")
    
    for opt in selected_opts:
        if "extruder temperature" in opt:
            new_temp = float(opt.split("to ")[1].split("°C")[0])
            for i, line in enumerate(optimized_lines):
                if line.startswith("M104 S") or line.startswith("M109 S"):
                    optimized_lines[i] = re.sub(r'S\d+\.?\d*', f'S{new_temp}', line)
                    break
        elif "bed temperature" in opt:
            new_temp = float(opt.split("to ")[1].split("°C")[0])
            for i, line in enumerate(optimized_lines):
                if line.startswith("M140 S") or line.startswith("M190 S"):
                    optimized_lines[i] = re.sub(r'S\d+\.?\d*', f'S{new_temp}', line)
                    break
        elif "travel speed" in opt:
            new_speed = float(opt.split("to ")[1].split(" mm/s")[0]) * 60
            for i, line in enumerate(optimized_lines):
                if line.startswith("G0 F"):
                    optimized_lines[i] = re.sub(r'F\d+\.?\d*', f'F{new_speed}', line)
                    break
    return optimized_lines

# Main Streamlit app
def main():
    st.title("G-code Optimizer for FDM 3D Printing")
    st.write("Upload a G-code file to visualize the print structure and optimize settings and paths.")

    material = st.selectbox("Select Material", list(material_settings.keys()))
    uploaded_file = st.file_uploader("Upload G-code File", type=["gcode", "txt"])

    if uploaded_file is not None:
        temp_file_path = "temp_gcode.gcode"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        gcode_str = uploaded_file.getvalue().decode("utf-8")
        gcode_lines = gcode_str.split('\n')
        settings = parse_gcode(gcode_str)
        reader = GcodeReader(filename=temp_file_path, filetype=GcodeType.FDM_REGULAR)

        # Initial layer visualization
        st.write("### Layer Visualization")
        try:
            if reader.n_layers > 0:
                layer_num = st.slider("Select Layer for Preview", 1, reader.n_layers, 1)
                fig = reader.plot_layer(layer_num)
                st.pyplot(fig)
            else:
                st.write("No layers detected in the G-code.")
        except Exception as e:
            st.error(f"Error visualizing G-code: {str(e)}")

        # Optimization suggestions
        recommended = material_settings[material]
        suggestions = ["Optimize travel paths"]
        if 'extruder_temp' in settings:
            if settings['extruder_temp'] < recommended['extruder_temp'][0]:
                suggestions.append(f"Increase extruder temperature from {settings['extruder_temp']}°C to {recommended['extruder_temp'][0]}°C")
            elif settings['extruder_temp'] > recommended['extruder_temp'][1]:
                suggestions.append(f"Decrease extruder temperature from {settings['extruder_temp']}°C to {recommended['extruder_temp'][1]}°C")
        if 'bed_temp' in settings:
            if settings['bed_temp'] < recommended['bed_temp'][0]:
                suggestions.append(f"Increase bed temperature from {settings['bed_temp']}°C to {recommended['bed_temp'][0]}°C")
            elif settings['bed_temp'] > recommended['bed_temp'][1]:
                suggestions.append(f"Decrease bed temperature from {settings['bed_temp']}°C to {recommended['bed_temp'][1]}°C")
        if 'travel_speed' in settings and settings['travel_speed'] < 150:
            suggestions.append(f"Increase travel speed from {settings['travel_speed']} mm/s to 150 mm/s")

        if suggestions:
            st.write("### Suggested Optimizations")
            selected_opts = []
            for sug in suggestions:
                if st.checkbox(sug, key=sug):
                    selected_opts.append(sug)

            if selected_opts and st.button("Optimize"):
                with st.spinner("Optimizing and generating path images..."):
                    optimized_lines = apply_optimizations(gcode_lines, selected_opts, reader)
                    original_reader = GcodeReader(gcode_lines=gcode_lines)
                    optimized_reader = GcodeReader(gcode_lines=optimized_lines)
                    st.session_state.optimized_lines = optimized_lines
                    st.session_state.original_reader = original_reader
                    st.session_state.optimized_reader = optimized_reader
                    st.session_state.optimization_done = True

        if 'optimization_done' in st.session_state and st.session_state.optimization_done:
            st.write("### Path Comparison")
            comparison_layer = st.slider("Select Layer for Comparison", 1, st.session_state.original_reader.n_layers, 1)
            col1, col2 = st.columns(2)
            with col1:
                original_fig = st.session_state.original_reader.plot_layer_path(comparison_layer)
                original_fig.suptitle(f"Original G-code Path (Layer {comparison_layer})")
                st.pyplot(original_fig)
            with col2:
                optimized_fig = st.session_state.optimized_reader.plot_layer_path(comparison_layer)
                optimized_fig.suptitle(f"Optimized G-code Path (Layer {comparison_layer})")
                st.pyplot(optimized_fig)
            
            st.download_button(
                label="Download Optimized G-code",
                data="\n".join(st.session_state.optimized_lines),
                file_name="optimized.gcode",
                mime="text/plain"
            )

        os.remove(temp_file_path)

if __name__ == "__main__":
    main()