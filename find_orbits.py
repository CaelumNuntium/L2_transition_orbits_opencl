import subprocess
import argparse
import os
import numpy as np
from math import *
from scipy import optimize
from astropy.io import fits
from matplotlib import pyplot as plt

integrator_directory = "./x64/Release"


class CoordinateArray(object):
    def __init__(self, data, delta_t):
        if data is not None:
            if data.shape[2] != 6:
                raise ValueError(f"Invalid data shape {data.shape}")
            self.data = np.array(data)
            self.delta_t = delta_t
            self.length = self.data.shape[0]
            self.t_max = (self.length - 1) * delta_t
            self.n_bodies = self.data.shape[1]
        else:
            self.data = None
            self.delta_t = None
            self.length = None
            self.t_max = None
            self.n_bodies = None

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def get_body(self, body_id):
        return self.__class__(self.data[:, body_id:(body_id + 1), :], self.delta_t)

    def save(self, filename, file_type):
        if file_type == "FITS":
            hdu = fits.PrimaryHDU(self.data)
            hdu.header["T_STEP"] = str(self.delta_t)
            hdu.writeto(filename, overwrite=True)
        elif file_type == "TEXT":
            raise ValueError("Text files are not supported in this version")
        else:
            raise ValueError(f"Invalid type of file: {file_type}")

    @classmethod
    def from_file(cls, filename, file_type):
        if file_type == "FITS":
            fits_file = fits.open(filename)
            data = fits_file[0].data
            delta_t = float(fits_file[0].header["T_STEP"])
            if data.shape[2] != 6:
                raise ValueError(f"Invalid data shape {data.shape} in {filename}")
            fits_file.close()
            return cls(data, delta_t)
        elif file_type == "TEXT":
            raise ValueError("Text files are not supported in this version")
        else:
            raise ValueError(f"Invalid type of file: {file_type}")


class CartesianCoordinateArray(CoordinateArray):
    def get_coordinate_array(self, coord):
        dict_coords = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5}
        return self.data[:, :, dict_coords[coord]]


class UnitName(object):
    def __init__(self, name, russian_name):
        self.name = name
        self.russian_name = russian_name

    def __repr__(self):
        if russian:
            return self.russian_name
        else:
            return self.name

    def __str__(self):
        return self.__repr__()


def jacobi_constant(x, y, z, vx, vy, vz):
    mu1 = 1.0 - mu2
    v_sq = vx * vx + vy * vy + vz * vz
    r1 = np.sqrt((x + mu2) * (x + mu2) + y * y + z * z)
    r2 = np.sqrt((x - mu1) * (x - mu1) + y * y + z * z)
    return x * x + y * y + (mu1 / r1 + mu2 / r2) * 2.0 - v_sq


def plot_jacobi_constant(t_arr, x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr, filename,
                         plot_difference=True, absolute_value=False, show=False, borders=None):
    plt.xlabel(f"$t$, {years_unit}")
    t_markers = t_arr / (2.0 * pi)
    c_j_arr = jacobi_constant(x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr)
    if plot_difference:
        c_j_0 = c_j_arr[0]
        print(f"C_J(0) = {c_j_0}")
        if absolute_value:
            plt.ylabel("$\\Delta C_J$")
            delim = 1.0
        else:
            plt.ylabel("$\\Delta C_J / C_J(0)$")
            delim = c_j_0
    else:
        c_j_0 = 0.0
        plt.ylabel("$C_J$")
        delim = 1.0
    u_lim = min([1.0E-4, 1.25 * np.max((c_j_arr - c_j_0) / delim)])
    l_lim = min([max([-1.0E-4, 1.25 * np.min((c_j_arr - c_j_0) / delim)]), -0.1 * u_lim])
    shift = 0.05 * (u_lim - l_lim)
    plt.ylim([l_lim, u_lim])
    plt.plot(t_markers, (c_j_arr - c_j_0) / delim, linewidth=0.8, color="green")
    if borders is not None:
        for b in borders:
            plt.plot([b / (2.0 * pi), b / (2.0 * pi)], [u_lim - shift, l_lim + shift], color="red", linewidth=0.4)
    plt.gca().invert_xaxis()
    f = plt.gcf()
    f.savefig(filename, dpi=450)
    if show:
        plt.show()
    plt.cla()
    plt.clf()


def read_config(filename: str):
    params = {}
    with open(filename, "r") as inp:
        for ss in inp:
            if '#' in ss:
                s1 = ss.split('#')[0]
            else:
                s1 = ss
            if '=' in s1:
                parameter = s1.split('=')[0].strip()
                value = s1.split('=')[1].strip()
                params[parameter] = value
    return params


class IntegratorParameters(object):
    def __init__(self, num_steps: int, step: float, m: int, n_bodies: int, model: str, data_file: str,
                 result_file: str, t0=0.0, output="binary", method="collo", **kwargs):
        self.num_steps = num_steps
        self.t0 = t0
        self.step = step
        self.m = m
        self.n_bodies = n_bodies
        self.model = model
        self.data_file = data_file
        self.result_file = result_file
        self.output = output
        self.method = method
        if self.method == "collo":
            if "collo_log2s" in kwargs.keys():
                self.collo_log2s = kwargs["collo_log2s"]
            else:
                self.collo_log2s = global_collo_log2s
        if "platform" in kwargs.keys():
            self.platform = kwargs["platform"]
        else:
            self.platform = global_platform
        if "device" in kwargs.keys():
            self.device = kwargs["device"]
        else:
            self.device = global_device
        self.status = "NOT SAVED"
        self.file = None

    def save(self, filename: str):
        with open(filename, "w") as config_out:
            config_out.write(f"num_steps = {self.num_steps}\n")
            config_out.write(f"t0 = {self.t0}\n")
            config_out.write(f"step = {self.step}\n")
            config_out.write(f"M_steps = {self.m}\n")
            config_out.write(f"N = {self.n_bodies}\n")
            config_out.write(f"equation = {self.model}\n")
            config_out.write(f"method = {self.method}\n")
            if self.method == "collo":
                config_out.write(f"log2S = {self.collo_log2s}\n")
            config_out.write(f"data_file = {self.data_file}\n")
            config_out.write(f"output = {self.output}\n")
            config_out.write(f"result_file = {self.result_file}\n")
            config_out.write(f'platform = "{self.platform}"\ndevice = "{self.device}"\n')
        self.status = "SAVED"
        self.file = filename

    @classmethod
    def load(cls, config_name: str):
        c_params = read_config(config_name)
        return cls(int(c_params["num_steps"]), float(c_params["step"]), int(c_params["M_steps"]), int(c_params["N"]),
                   c_params["equation"], c_params["data_file"], c_params["result_file"], t0=float(c_params["t0"]),
                   output=c_params["output"], method=c_params["method"])


class Integrator(object):
    def __init__(self, config_name: str):
        self.config_file = config_name
        self.parameters = IntegratorParameters.load(config_name)
        self.exec_file = f"{integrator_directory}/integrator.exe"
        self.status = "NOT STARTED"
        self.process = None
        self.exit_code = None
        self.result = CartesianCoordinateArray(None, None)

    def start(self, mute=False):
        self.status = "IN PROGRESS"
        if mute:
            self.process = subprocess.Popen([self.exec_file, self.config_file], stdout=subprocess.DEVNULL)
        else:
            self.process = subprocess.Popen([self.exec_file, self.config_file])
        self.exit_code = self.process.wait()
        if self.exit_code != 0:
            self.status = "ERROR"
        else:
            self.status = "FINISHED"
        if self.parameters.output == "binary":
            file_type = "FITS"
        else:
            file_type = "TEXT"
        self.result = CartesianCoordinateArray.from_file(self.parameters.result_file, file_type)

    def set_result_without_integrating(self):
        if self.parameters.output == "binary":
            file_type = "FITS"
        else:
            file_type = "TEXT"
        self.result = CartesianCoordinateArray.from_file(self.parameters.result_file, file_type)


def earth_coords(t):
    theta = t * n_l
    return 1.0 - mu2 + ll * delta * np.cos(theta), ll * delta * np.sin(theta)


def moon_coords(t):
    theta = t * n_l
    return 1.0 - mu2 - ll * (1.0 - delta) * np.cos(theta), -ll * (1.0 - delta) * np.sin(theta)


def earth_velocity(t):
    theta = t * n_l
    return -ll * delta * n_l * np.sin(theta), ll * delta * n_l * np.cos(theta)


def moon_velocity(t):
    theta = t * n_l
    return ll * (1.0 - delta) * n_l * np.sin(theta), -ll * (1.0 - delta) * n_l * np.cos(theta)


def stage1(model="restricted_3body", stage_number=1):
    print(f"\n***** STAGE {stage_number} *****\n")
    stage_dir = f"stage{stage_number}"
    os.makedirs(f"{data_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{config_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{result_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{log_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{img_dir}/{stage_dir}", exist_ok=True)
    m0 = 10
    n0_bodies = 2000
    v_range_len = 200 / mps_in_au_per_t_unit
    vy0 = vy0_init - 0.5 * v_range_len
    t_max = 2000
    steps_in_t_unit = 100
    t_unit = 0.01
    t_step = t_unit / m0
    with open(f"stage{stage_number}.log", "w") as out:
        i_step = 0
        ok = False
        while not ok:
            dvy = v_range_len / n0_bodies
            i_step = i_step + 1
            print(f"\nv_range_len = {v_range_len} AU / t_unit = {v_range_len * mps_in_au_per_t_unit} m / s")
            print(f"dvy = {dvy} AU / t_unit = {dvy * mps_in_au_per_t_unit} m / s")
            print(f"t_max = {t_max / steps_in_t_unit / (2 * pi)} years\n")
            out.write(f"Step {i_step}:\n")
            out.write(f"v_range_len = {v_range_len} AU / year = {v_range_len * mps_in_au_per_t_unit} m / s\n")
            out.write(f"dvy = {dvy} AU / year = {dvy * mps_in_au_per_t_unit} m / s\n")
            out.write(f"t_max = {t_max / steps_in_t_unit / (2 * pi)} years\n")
            d_name = f"{data_dir}/{stage_dir}/data_{round(log10(v_range_len * mps_in_au_per_t_unit))}.dat"
            conf_name = f"{config_dir}/{stage_dir}/config_{round(log10(v_range_len * mps_in_au_per_t_unit))}.config"
            res_name = f"{result_dir}/{stage_dir}/result_{round(log10(v_range_len * mps_in_au_per_t_unit))}.fits"
            log_name = f"{log_dir}/{stage_dir}/log_{round(log10(v_range_len * mps_in_au_per_t_unit))}.log"
            if integrate:
                with open(d_name, "w") as d_out:
                    for i in range(n0_bodies):
                        d_out.write(f"{x0:.20f} 0.0 0.0 0.0 {vy0 + i * dvy:.20f} 0.0\n")
                stage1_integrator_params = IntegratorParameters(t_max * m0, -t_step, m0, n0_bodies, model, d_name,
                                                                res_name)
                stage1_integrator_params.save(conf_name)
                integrator = Integrator(conf_name)
                integrator.start()
                if integrator.exit_code != 0:
                    raise RuntimeError("Error in integrator!")
            else:
                integrator = Integrator(conf_name)
                integrator.set_result_without_integrating()
            coord_arr = integrator.result
            x_lim = np.zeros(n0_bodies)
            y_lim = np.zeros(n0_bodies)
            z_lim = np.zeros(n0_bodies)
            vx_lim = np.zeros(n0_bodies)
            vy_lim = np.zeros(n0_bodies)
            vz_lim = np.zeros(n0_bodies)
            t_lim = np.zeros(n0_bodies)
            rd = round(abs(log10(dvy))) + 1
            stable_lim1 = vy0
            stable_lim2 = vy0
            stable_id1 = 0
            stable_id2 = 0
            is_stable = False
            with open(log_name, "w") as log_out:
                log_out.write(f"lim1 = {lim1}; lim2 = {lim2}\n")
                for body_id in range(n0_bodies):
                    x_arr = coord_arr.get_body(body_id).get_coordinate_array("x")[0:t_max, 0]
                    y_arr = coord_arr.get_body(body_id).get_coordinate_array("y")[0:t_max, 0]
                    z_arr = coord_arr.get_body(body_id).get_coordinate_array("z")[0:t_max, 0]
                    vx_arr = coord_arr.get_body(body_id).get_coordinate_array("vx")[0:t_max, 0]
                    vy_arr = coord_arr.get_body(body_id).get_coordinate_array("vy")[0:t_max, 0]
                    vz_arr = coord_arr.get_body(body_id).get_coordinate_array("vz")[0:t_max, 0]
                    ier = 0
                    for jj in range(len(x_arr)):
                        x = x_arr[jj]
                        y = y_arr[jj]
                        z = z_arr[jj]
                        vx = vx_arr[jj]
                        vy = vy_arr[jj]
                        vz = vz_arr[jj]
                        if x < lim1 or x > lim2:
                            log_out.write(f"Body {body_id}: vy0 = {round(vy0 + body_id * dvy, rd)} - x_lim = {x}\n")
                            x_lim[body_id] = x
                            y_lim[body_id] = y
                            z_lim[body_id] = z
                            vx_lim[body_id] = vx
                            vy_lim[body_id] = vy
                            vz_lim[body_id] = vz
                            t_lim[body_id] = jj
                            ier = 1
                            break
                    if ier == 0:
                        log_out.write(f"Body {body_id}: vy0 = {round(vy0 + body_id * dvy, rd)} - "
                                      f"x_lim = {x_arr[t_max - 1]} (STABLE)\n")
                        x_lim[body_id] = x_arr[t_max - 1]
                        y_lim[body_id] = y_arr[t_max - 1]
                        z_lim[body_id] = z_arr[t_max - 1]
                        vx_lim[body_id] = vx_arr[t_max - 1]
                        vy_lim[body_id] = vy_arr[t_max - 1]
                        vz_lim[body_id] = vz_arr[t_max - 1]
                        t_lim[body_id] = t_max - 1
                        if not is_stable:
                            stable_lim1 = vy0 + body_id * dvy
                            stable_id1 = body_id
                            is_stable = True
                        else:
                            stable_lim2 = vy0 + body_id * dvy
                            stable_id2 = body_id
                    else:
                        if body_id > 0:
                            if x_lim[body_id - 1] < lim1 and x_lim[body_id] > lim2:
                                stable_id1 = body_id - 1
                                stable_id2 = body_id
                                stable_lim1 = vy0 + stable_id1 * dvy
                                stable_lim2 = vy0 + stable_id2 * dvy
            print(f"Stable orbits: from vy = {stable_lim1} (id = {stable_id1}) to vy = {stable_lim2} "
                  f"(id = {stable_id2})\n")
            out.write(f"Stable orbits: from vy = {stable_lim1} (id = {stable_id1}) to vy = {stable_lim2} "
                      f"(id = {stable_id2})\n\n")
            if russian:
                plt.xlabel("$v_{y0}$, м / с")
                plt.ylabel("$x$, а. е.")
            else:
                plt.xlabel("$v_{y0}$, m / s")
                plt.ylabel("$x$, AU")
            plt.plot([(vy0 + _ * dvy) * mps_in_au_per_t_unit for _ in range(n0_bodies)], x_lim, color="blue")
            plt.plot([stable_lim1 * mps_in_au_per_t_unit, stable_lim1 * mps_in_au_per_t_unit], [lim1, lim2],
                     color="red")
            plt.plot([stable_lim2 * mps_in_au_per_t_unit, stable_lim2 * mps_in_au_per_t_unit], [lim1, lim2],
                     color="red")
            fig = plt.gcf()
            fig.savefig(f"{img_dir}/{stage_dir}/plot_{round(log10(v_range_len * mps_in_au_per_t_unit))}.png", dpi=300)
            plt.cla()
            plt.clf()
            if dvy < 1.0E-17:
                best_vy0 = vy0
                best_dvy = dvy
                with open(f"{data_dir}/{stage_dir}/trajectories.dat", "w") as new_data:
                    for jj in range(stable_id1):
                        new_data.write(f"{x_lim[jj]:.24f} {y_lim[jj]:.24f} {z_lim[jj]:.24f} {vx_lim[jj]:.24f} "
                                       f"{vy_lim[jj]:.24f} {vz_lim[jj]:.24f}\n")
                with open(f"{data_dir}/{stage_dir}/times.dat", "w") as time_data:
                    for jj in range(stable_id1):
                        time_data.write(f"{t_lim[jj]}\n")
                best_id = int((stable_id2 - stable_id1) / 2) + stable_id1
                best_vy = vy0 + best_id * dvy
                max_time = t_max
                print(f"Best orbit is {best_id}: vy = {vy0 + dvy * best_id} = "
                      f"{(vy0 + dvy * best_id) * mps_in_au_per_t_unit} m/s")
                out.write(f"\nBest orbit is {best_id}: vy = {vy0 + dvy * best_id} = "
                          f"{(vy0 + dvy * best_id) * mps_in_au_per_t_unit} m/s\n")
                best_x = coord_arr.get_body(best_id).get_coordinate_array("x")[0:t_max, 0]
                best_y = coord_arr.get_body(best_id).get_coordinate_array("y")[0:t_max, 0]
                max_stable_time = 0
                for j in range(len(best_y) - 1):
                    if best_x[j] < lim1 or best_x[j] > lim2:
                        max_stable_time = j - 1
                        break
                print(f"Best orbit is stable in {max_stable_time / steps_in_t_unit / (2 * pi)} years")
                out.write(f"Best orbit is stable in {max_stable_time / steps_in_t_unit / (2 * pi)} years\n")
                best_x = best_x[0:(max_stable_time + 1)]
                best_y = best_y[0:(max_stable_time + 1)]
                plt.gca().set_aspect('equal', adjustable='box')
                if russian:
                    plt.xlabel("$x$, $10^6$ км")
                    plt.ylabel("$y$, $10^6$ км")
                else:
                    plt.xlabel("$x$, $10^6$ km")
                    plt.ylabel("$y$, $10^6$ km")
                plt.plot((best_x - x_l2) * km_in_au / 1000000, best_y * km_in_au / 1000000, color="green")
                plt.plot([(lim1 - x_l2) * km_in_au / 1000000, (lim1 - x_l2) * km_in_au / 1000000], [-0.7, 0.7],
                         color="red", linewidth=0.5)
                plt.plot([(lim2 - x_l2) * km_in_au / 1000000, (lim2 - x_l2) * km_in_au / 1000000], [-0.7, 0.7],
                         color="red", linewidth=0.5)
                plt.scatter(0.0, 0.0, marker="o", color="red", s=5)
                plt.scatter((x_earth - x_l2) * km_in_au / 1000000, 0.0, marker="o", color="blue", s=6)
                fig = plt.gcf()
                fig.savefig(f"{img_dir}/{stage_dir}/best_orbit_xy.png", dpi=600)
                plt.cla()
                plt.clf()
                ok = True
            if stable_id2 - stable_id1 == 1:
                v_range_len = 2 * dvy
                vy0 = stable_lim1 - dvy
            else:
                v_range_len = stable_lim2 - stable_lim1
                vy0 = stable_lim1
        with open(f"{log_dir}/{stage_dir}/stage{stage_number}_log.log", "w") as lout:
            lout.write(f"vy0 = {best_vy0}\n")
            lout.write(f"dvy = {best_dvy}\n")
            lout.write(f"n_trajectories = {stable_id1}\n")
            lout.write(f"max_time = {max_time}\n")
            lout.write(f"best_vy = {best_vy}\n")


def stage2():
    stage1(model="bicircular_r4bp", stage_number=2)


def stage3(model="restricted_3body", stage_number=3):
    print(f"\n***** STAGE {stage_number} *****\n")
    stage_dir = f"stage{stage_number}"
    os.makedirs(f"{data_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{config_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{result_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{log_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{img_dir}/{stage_dir}", exist_ok=True)
    with open(f"{log_dir}/{stage_dir}/stage{stage_number}.log", "w") as log_out:
        stage1_params = read_config(f"{log_dir}/stage{stage_number - 2}/stage{stage_number - 2}_log.log")
        max_time = int(stage1_params["max_time"])
        best_vy = float(stage1_params["best_vy"])
        n_trajectories = 2000
        m0 = 10
        t_unit = 0.01
        t_step = t_unit / m0
        dvy = 5.0E-17
        vy_start = best_vy - n_trajectories * dvy
        log_out.write(f"t_unit_1 = {t_unit}\n")
        d_name = f"{data_dir}/{stage_dir}/trajectories.dat"
        res_name = f"{result_dir}/{stage_dir}/trajectories.fits"
        conf_name = f"{config_dir}/{stage_dir}/trajectories.config"
        if integrate:
            with open(d_name, "w") as data_out:
                for j in range(n_trajectories):
                    data_out.write(f"{x0:.20f} 0.0 0.0 0.0 {vy_start + j * dvy:.20f} 0.0\n")
            stage3_integrator_params = IntegratorParameters(max_time * m0, -t_step, m0, n_trajectories, model, d_name,
                                                            res_name)
            stage3_integrator_params.save(conf_name)
            integrator = Integrator(conf_name)
            integrator.start()
            if integrator.exit_code != 0:
                raise RuntimeError("Error in integrator!")
        else:
            integrator = Integrator(conf_name)
            integrator.set_result_without_integrating()
        coord_arr = integrator.result
        d1_name = f"{data_dir}/{stage_dir}/earth_trajectories.dat"
        with open(d1_name, "w") as data_out:
            t_lim = np.zeros(n_trajectories)
            for body_id in range(n_trajectories):
                x_arr = coord_arr.get_body(body_id).get_coordinate_array("x")[0:max_time, 0]
                ier = 0
                for jj in range(len(x_arr)):
                    x = x_arr[jj]
                    if x < lim1 or x > lim2:
                        t_lim[body_id] = jj
                        ier = 1
                        break
                if ier == 0:
                    t_lim[body_id] = max_time - 1
            t_limit = int(np.min(t_lim))
            log_out.write(f"t_limit = {t_limit}\n")
            t_lim_max = int(np.max(t_lim))
            for body_id in range(n_trajectories):
                x_arr = coord_arr.get_body(body_id).get_coordinate_array("x")[0:max_time, 0]
                y_arr = coord_arr.get_body(body_id).get_coordinate_array("y")[0:max_time, 0]
                vx_arr = coord_arr.get_body(body_id).get_coordinate_array("vx")[0:max_time, 0]
                vy_arr = coord_arr.get_body(body_id).get_coordinate_array("vy")[0:max_time, 0]
                data_out.write(f"{x_arr[t_limit]:.20f} {y_arr[t_limit]:20f} 0.0 {vx_arr[t_limit]:.20f} "
                               f"{vy_arr[t_limit]:.20f} 0.0\n")
        t_n_steps = 150000 + 2 * 100 * (t_lim_max - t_limit)
        t_step = 0.00002
        m1 = 5
        t_n_units = t_n_steps // m1
        t_unit1 = t_step * m1
        log_out.write(f"t_n_steps_2 = {t_n_steps}\n")
        log_out.write(f"t_unit_2 = {t_step * m1}\n")
        conf1_name = f"{config_dir}/{stage_dir}/earth_trajectories.config"
        res1_name = f"{result_dir}/{stage_dir}/earth_trajectories.fits"
        if integrate:
            stage3_integrator_params1 = IntegratorParameters(t_n_steps, -t_step, m1, n_trajectories, model, d1_name,
                                                             res1_name, t0=-t_limit * t_unit)
            stage3_integrator_params1.save(conf1_name)
            integrator = Integrator(conf1_name)
            integrator.start()
            if integrator.exit_code != 0:
                raise RuntimeError("Error in integrator!")
        else:
            integrator = Integrator(conf1_name)
            integrator.set_result_without_integrating()
        coord_arr1 = integrator.result
        earth_distances = []
        moon_distances = []
        earth_positions = []
        moon_positions = []
        ref_times = []
        t_marks = np.array([-t_limit * t_unit - _ * t_step * m1 for _ in range(t_n_units)])
        if model == "bicircular_r4bp":
            x_earth_arr, y_earth_arr = earth_coords(t_marks)
            x_moon_arr, y_moon_arr = moon_coords(t_marks)
        else:
            x_earth_arr = np.ones(t_n_units) * x_earth
            y_earth_arr = np.zeros(t_n_units)
        for bid in range(n_trajectories):
            coords = coord_arr1.get_body(bid)
            x_arr = coords.get_coordinate_array("x")[0:t_n_units, 0]
            y_arr = coords.get_coordinate_array("y")[0:t_n_units, 0]
            earth_dist_sq_arr = (x_arr - x_earth_arr) * (x_arr - x_earth_arr) + (y_arr - y_earth_arr) * (y_arr -
                                                                                                         y_earth_arr)
            earth_distances.append(sqrt(np.min(earth_dist_sq_arr)))
            ref_time = np.argmin(earth_dist_sq_arr)
            impact = earth_dist_sq_arr < r_earth_sq
            if np.any(impact):
                impact_time = np.argmax(impact)
            else:
                impact_time = t_n_steps
            if impact_time < ref_time:
                ref_time = impact_time
            ref_times.append(ref_time)
            earth_positions.append(
                (x_earth_arr[np.argmin(earth_dist_sq_arr)], y_earth_arr[np.argmin(earth_dist_sq_arr)]))
            if model == "bicircular_r4bp":
                moon_dist_sq_arr = (x_arr - x_moon_arr) * (x_arr - x_moon_arr) + (y_arr - y_moon_arr) * (y_arr -
                                                                                                         y_moon_arr)
                moon_distances.append(sqrt(np.min(moon_dist_sq_arr[0:ref_time])))
                moon_positions.append((x_moon_arr[np.argmin(moon_dist_sq_arr[0:ref_time])],
                                       y_moon_arr[np.argmin(moon_dist_sq_arr[0:ref_time])]))
        earth_distances = np.array(earth_distances)
        moon_distances = np.array(moon_distances)
        min_earth_dist_id = np.argmin(earth_distances)
        min_earth_dist_dvy = (min_earth_dist_id - n_trajectories) * dvy * mps_in_au_per_t_unit * 1.0E9
        max_earth_dist_id = np.argmax(earth_distances[:-1])
        max_earth_dist_dvy = (max_earth_dist_id - n_trajectories) * dvy * mps_in_au_per_t_unit * 1.0E9
        interesting1_id = np.argmax(earth_distances[:-100])
        interesting1_dvy = (interesting1_id - n_trajectories) * dvy * mps_in_au_per_t_unit * 1.0E9
        if model == "restricted_3body":
            log_out.write(f"min_earth_dist_id = {min_earth_dist_id}\n")
            log_out.write(f"max_earth_dist_id = {max_earth_dist_id}\n")
            log_out.write(f"interesting1_id = {interesting1_id}\n")
        if model == "bicircular_r4bp":
            min_moon_dist_id = np.argmin(moon_distances)
            log_out.write(f"min_moon_dist_id = {min_moon_dist_id}\n")
            # min_moon_dist_dvy = (min_moon_dist_id - n_trajectories) * dvy * mps_in_au_per_t_unit * 1.0E9
        if model == "bicircular_r4bp":
            min_good_moon_distance = 200000 / km_in_au
            if russian:
                plt.xlabel("$\\varepsilon = v_{y0} - v^*_{y0}$, $10^{-9}$ м / с")
                plt.ylabel("$r_{2, min}$, $r_{3, min}$, км")
            else:
                plt.xlabel("$\\varepsilon = v_{y0} - v^*_{y0}$, $10^{-9}$ m / s")
                plt.ylabel("$r_{2, min}$, $r_{3, min}$, km")
            plt.ylim((-10000.0, 800000.0))
            is_good = moon_distances > min_good_moon_distance
            good_earth_distances = []
            good_vys = []
            bad_earth_distances = []
            bad_vys = []
            all_good_earth_distances = []
            all_good_vys = []
            all_good_ids = []
            for jj in range(len(is_good)):
                if is_good[jj]:
                    bad_earth_distances.append(earth_distances[jj])
                    bad_vys.append((jj - n_trajectories) * dvy)
                    plt.plot(np.array(bad_vys) * mps_in_au_per_t_unit * 1.0E9, np.array(bad_earth_distances) * km_in_au,
                             color="red", linewidth=0.25, linestyle="dashed")
                    bad_earth_distances = []
                    bad_vys = []
                    good_earth_distances.append(earth_distances[jj])
                    all_good_earth_distances.append(earth_distances[jj])
                    good_vys.append((jj - n_trajectories) * dvy)
                    all_good_vys.append((jj - n_trajectories) * dvy)
                    all_good_ids.append(jj)
                else:
                    plt.plot(np.array(good_vys) * mps_in_au_per_t_unit * 1.0E9,
                             np.array(good_earth_distances) * km_in_au, color="red", linewidth=1.0)
                    good_earth_distances = []
                    good_vys = []
                    bad_earth_distances.append(earth_distances[jj - 1])
                    bad_vys.append((jj - 1 - n_trajectories) * dvy)
                    bad_earth_distances.append(earth_distances[jj])
                    bad_vys.append((jj - n_trajectories) * dvy)
            all_good_earth_distances = np.array(all_good_earth_distances)
            # all_good_vys = np.array(all_good_vys)
            # plt.plot(all_good_vys * mps_in_au_per_t_unit * 1.0E9, all_good_earth_distances * km_in_au, color="red",
            #          linewidth=1.0)
            min_earth_dist_id = all_good_ids[np.argmin(all_good_earth_distances)]
            log_out.write(f"min_earth_dist_id = {min_earth_dist_id}\n")
            max_earth_dist_id = all_good_ids[np.argmax(all_good_earth_distances)]
            log_out.write(f"max_earth_dist_id = {max_earth_dist_id}\n")
            interesting1_id = all_good_ids[np.argmax(all_good_earth_distances[:-200])]
            log_out.write(f"interesting1_id = {interesting1_id}\n")
            min_earth_dist_dvy = (min_earth_dist_id - n_trajectories) * dvy * mps_in_au_per_t_unit * 1.0E9
            max_earth_dist_dvy = (max_earth_dist_id - n_trajectories) * dvy * mps_in_au_per_t_unit * 1.0E9
            interesting1_dvy = (interesting1_id - n_trajectories) * dvy * mps_in_au_per_t_unit * 1.0E9
            # plt.plot(np.array(good_vys) * mps_in_au_per_t_unit * 1.0E9, np.array(good_earth_distances) * km_in_au,
            #          color="red", linewidth=1.0)
            plt.plot([((_ - n_trajectories) * dvy) * mps_in_au_per_t_unit * 1.0E9 for _ in range(n_trajectories)],
                     moon_distances * km_in_au, color="blue", linewidth=0.4)
            plt.plot([-3.0, 0.0], [min_good_moon_distance * km_in_au, min_good_moon_distance * km_in_au], color="gray",
                     linewidth=0.5)
            plt.plot([min_earth_dist_dvy, min_earth_dist_dvy], [0, 750000], color="cyan", linewidth=0.85)
            plt.plot([max_earth_dist_dvy, max_earth_dist_dvy], [0, 750000], color="pink", linewidth=0.85)
            plt.plot([interesting1_dvy, interesting1_dvy], [0, 750000], color="purple", linewidth=0.85)
            # plt.plot([min_moon_dist_dvy, min_moon_dist_dvy], [0, 750000], color="gray", linewidth=0.85)
            fig = plt.gcf()
            fig.savefig(f"{img_dir}/{stage_dir}/find_trajectories_moon.png", dpi=600)
            plt.cla()
            plt.clf()
        if russian:
            plt.xlabel("$\\varepsilon = v_{y0} - v^*_{y0}$, $10^{-9}$ м / с")
            plt.ylabel("$r_{2, min}$, км")
        else:
            plt.xlabel("$\\varepsilon = v_{y0} - v^*_{y0}$, $10^{-9}$ m / s")
            plt.ylabel("$r_{2, min}$, km")
        plt.ylim((-10000.0, 800000.0))
        plt.plot([((_ - n_trajectories) * dvy) * mps_in_au_per_t_unit * 1.0E9 for _ in range(n_trajectories)],
                 earth_distances * km_in_au, color="green", linewidth=1.0)
        if not model == "bicircular_r4bp":
            plt.plot([min_earth_dist_dvy, min_earth_dist_dvy], [0, 750000], color="cyan", linewidth=0.85)
            plt.plot([max_earth_dist_dvy, max_earth_dist_dvy], [0, 750000], color="pink", linewidth=0.85)
            plt.plot([interesting1_dvy, interesting1_dvy], [0, 750000], color="purple", linewidth=0.85)
        if model == "bicircular_r4bp":
            plt.plot([((_ - n_trajectories) * dvy) * mps_in_au_per_t_unit * 1.0E9 for _ in range(n_trajectories)],
                     moon_distances * km_in_au, color="blue", linewidth=0.6)
        fig = plt.gcf()
        fig.savefig(f"{img_dir}/{stage_dir}/find_trajectories.png", dpi=600)
        plt.cla()
        plt.clf()
        t_arr = np.array([-_ * t_unit for _ in range(t_limit)])
        t_arr1 = np.array([-t_limit * t_unit - _ * t_unit1 for _ in range(int(np.max(ref_times)))])
        min_earth_dist_x_arr = coord_arr.get_body(min_earth_dist_id).get_coordinate_array("x")[0:t_limit, 0]
        min_earth_dist_y_arr = coord_arr.get_body(min_earth_dist_id).get_coordinate_array("y")[0:t_limit, 0]
        min_earth_dist_vx_arr = coord_arr.get_body(min_earth_dist_id).get_coordinate_array("vx")[0:t_limit, 0]
        min_earth_dist_vy_arr = coord_arr.get_body(min_earth_dist_id).get_coordinate_array("vy")[0:t_limit, 0]
        min_earth_dist_x_arr1 = coord_arr1.get_body(min_earth_dist_id
                                                    ).get_coordinate_array("x")[0:int(ref_times[min_earth_dist_id]), 0]
        min_earth_dist_y_arr1 = coord_arr1.get_body(min_earth_dist_id
                                                    ).get_coordinate_array("y")[0:int(ref_times[min_earth_dist_id]), 0]
        min_earth_dist_vx_arr1 = coord_arr1.get_body(min_earth_dist_id
                                                     ).get_coordinate_array("vx")[0:int(ref_times[min_earth_dist_id]), 0]
        min_earth_dist_vy_arr1 = coord_arr1.get_body(min_earth_dist_id
                                                     ).get_coordinate_array("vy")[0:int(ref_times[min_earth_dist_id]), 0]
        min_earth_dist_x_arr = np.append(min_earth_dist_x_arr, min_earth_dist_x_arr1)
        min_earth_dist_y_arr = np.append(min_earth_dist_y_arr, min_earth_dist_y_arr1)
        min_earth_dist_vx_arr = np.append(min_earth_dist_vx_arr, min_earth_dist_vx_arr1)
        min_earth_dist_vy_arr = np.append(min_earth_dist_vy_arr, min_earth_dist_vy_arr1)
        max_earth_dist_x_arr = coord_arr.get_body(max_earth_dist_id).get_coordinate_array("x")[0:t_limit, 0]
        max_earth_dist_y_arr = coord_arr.get_body(max_earth_dist_id).get_coordinate_array("y")[0:t_limit, 0]
        max_earth_dist_vx_arr = coord_arr.get_body(max_earth_dist_id).get_coordinate_array("vx")[0:t_limit, 0]
        max_earth_dist_vy_arr = coord_arr.get_body(max_earth_dist_id).get_coordinate_array("vy")[0:t_limit, 0]
        max_earth_dist_x_arr1 = coord_arr1.get_body(max_earth_dist_id
                                                    ).get_coordinate_array("x")[0:int(ref_times[max_earth_dist_id]), 0]
        max_earth_dist_y_arr1 = coord_arr1.get_body(max_earth_dist_id
                                                    ).get_coordinate_array("y")[0:int(ref_times[max_earth_dist_id]), 0]
        max_earth_dist_vx_arr1 = coord_arr1.get_body(max_earth_dist_id
                                                     ).get_coordinate_array("vx")[0:int(ref_times[max_earth_dist_id]), 0]
        max_earth_dist_vy_arr1 = coord_arr1.get_body(max_earth_dist_id
                                                     ).get_coordinate_array("vy")[0:int(ref_times[max_earth_dist_id]), 0]
        max_earth_dist_x_arr = np.append(max_earth_dist_x_arr, max_earth_dist_x_arr1)
        max_earth_dist_y_arr = np.append(max_earth_dist_y_arr, max_earth_dist_y_arr1)
        max_earth_dist_vx_arr = np.append(max_earth_dist_vx_arr, max_earth_dist_vx_arr1)
        max_earth_dist_vy_arr = np.append(max_earth_dist_vy_arr, max_earth_dist_vy_arr1)
        # max_earth_dist_x_arr = coord_arr1.get_body(max_earth_dist_id).get_coordinate_array("x")[:, 0]
        # max_earth_dist_y_arr = coord_arr1.get_body(max_earth_dist_id).get_coordinate_array("y")[:, 0]
        interesting1_x_arr = coord_arr.get_body(interesting1_id).get_coordinate_array("x")[0:t_limit, 0]
        interesting1_y_arr = coord_arr.get_body(interesting1_id).get_coordinate_array("y")[0:t_limit, 0]
        interesting1_vx_arr = coord_arr.get_body(interesting1_id).get_coordinate_array("vx")[0:t_limit, 0]
        interesting1_vy_arr = coord_arr.get_body(interesting1_id).get_coordinate_array("vy")[0:t_limit, 0]
        interesting1_x_arr1 = coord_arr1.get_body(interesting1_id
                                                  ).get_coordinate_array("x")[0:int(ref_times[interesting1_id]) + 150, 0]
        interesting1_y_arr1 = coord_arr1.get_body(interesting1_id
                                                  ).get_coordinate_array("y")[0:int(ref_times[interesting1_id]) + 150, 0]
        interesting1_vx_arr1 = coord_arr1.get_body(interesting1_id
                                                   ).get_coordinate_array("vx")[0:int(ref_times[interesting1_id]) + 150, 0]
        interesting1_vy_arr1 = coord_arr1.get_body(interesting1_id
                                                   ).get_coordinate_array("vy")[0:int(ref_times[interesting1_id]) + 150, 0]
        interesting1_x_arr = np.append(interesting1_x_arr, interesting1_x_arr1)
        interesting1_y_arr = np.append(interesting1_y_arr, interesting1_y_arr1)
        interesting1_vx_arr = np.append(interesting1_vx_arr, interesting1_vx_arr1)
        interesting1_vy_arr = np.append(interesting1_vy_arr, interesting1_vy_arr1)
        # if model == "bicircular_r4bp":
        #     min_moon_dist_x_arr = coord_arr.get_body(min_moon_dist_id).get_coordinate_array("x")[0:t_limit, 0]
        #     min_moon_dist_y_arr = coord_arr.get_body(min_moon_dist_id).get_coordinate_array("y")[0:t_limit, 0]
        #     min_moon_dist_x_arr1 = coord_arr1.get_body(min_moon_dist_id
        #                                                ).get_coordinate_array("x")[0:int(ref_times[min_moon_dist_id]), 0]
        #     min_moon_dist_y_arr1 = coord_arr1.get_body(min_moon_dist_id
        #                                                ).get_coordinate_array("y")[0:int(ref_times[min_moon_dist_id]), 0]
        #     min_moon_dist_x_arr = np.append(min_moon_dist_x_arr, min_moon_dist_x_arr1)
        #     min_moon_dist_y_arr = np.append(min_moon_dist_y_arr, min_moon_dist_y_arr1)
        plt.gca().set_aspect('equal', adjustable='box')
        if russian:
            plt.xlabel("$x$, $10^6$ км")
            plt.ylabel("$y$, $10^6$ км")
        else:
            plt.xlabel("$x$, $10^6$ km")
            plt.ylabel("$y$, $10^6$ km")
        plt.plot((min_earth_dist_x_arr - x_earth) * km_in_au * 1.0E-6, min_earth_dist_y_arr * km_in_au * 1.0E-6,
                 color="cyan", linewidth=0.95)
        plt.plot((max_earth_dist_x_arr - x_earth) * km_in_au * 1.0E-6, max_earth_dist_y_arr * km_in_au * 1.0E-6,
                 color="pink", linewidth=0.95)
        plt.plot((interesting1_x_arr - x_earth) * km_in_au * 1.0E-6, interesting1_y_arr * km_in_au * 1.0E-6,
                 color="purple", linewidth=0.95)
        plt.scatter([0.0], [0.0], s=6, color="blue")
        plt.scatter([(x_l2 - x_earth) * km_in_au * 1.0E-6], [0.0], s=5, color="red")
        if model == "bicircular_r4bp":
            moon_coord_array = moon_coords(np.array([0.001 * _ * moon_period for _ in range(1001)]))
            plt.plot((moon_coord_array[0] - x_earth) * km_in_au * 1.0E-6, moon_coord_array[1] * km_in_au * 1.0E-6,
                     color="black", linewidth=0.2)
            # plt.plot((min_moon_dist_x_arr - x_earth) * km_in_au * 1.0E-6, min_moon_dist_y_arr * km_in_au * 1.0E-6,
            #          color="gray", linewidth=0.95)
            # plt.scatter([(moon_positions[min_moon_dist_id][0] - x_earth) * km_in_au * 1.0E-6],
            #             [moon_positions[min_moon_dist_id][1] * km_in_au * 1.0E-6], s=3, color="black")
        fig = plt.gcf()
        fig.savefig(f"{img_dir}/{stage_dir}/all_trajectories.png", dpi=600)
        plt.cla()
        plt.clf()
        min_earth_dist_t_arr = np.append(t_arr, t_arr1[0:int(ref_times[min_earth_dist_id])])
        max_earth_dist_t_arr = np.append(t_arr, t_arr1[0:int(ref_times[max_earth_dist_id])])
        interesting1_t_arr = np.append(t_arr, t_arr1[0:(int(ref_times[interesting1_id]) + 150)])
        border = t_arr[-1]
        plot_jacobi_constant(min_earth_dist_t_arr, min_earth_dist_x_arr, min_earth_dist_y_arr, 0.0,
                             min_earth_dist_vx_arr, min_earth_dist_vy_arr, 0.0,
                             f"{img_dir}/{stage_dir}/min_earth_dist_c_j.png", borders=[border])
        plot_jacobi_constant(max_earth_dist_t_arr, max_earth_dist_x_arr, max_earth_dist_y_arr, 0.0,
                             max_earth_dist_vx_arr, max_earth_dist_vy_arr, 0.0,
                             f"{img_dir}/{stage_dir}/max_earth_dist_c_j.png", borders=[border])
        plot_jacobi_constant(interesting1_t_arr, interesting1_x_arr, interesting1_y_arr, 0.0, interesting1_vx_arr,
                             interesting1_vy_arr, 0.0, f"{img_dir}/{stage_dir}/interesting1_c_j.png", borders=[border])


def stage4():
    stage3(model="bicircular_r4bp", stage_number=4)


class Map2D(object):
    def __init__(self, arr, center: tuple[float, float], xy_range: tuple[float, float], t_unit: float,
                 map_type="coords", xy_unit=UnitName("km", "км"), coord_array=CartesianCoordinateArray(None, None)):
        self.data = arr
        self.x_center = center[0]
        self.y_center = center[1]
        self.xrange = xy_range[0]
        self.yrange = xy_range[1]
        self.map_type = map_type
        self.shape = self.data.shape
        self.t_unit = t_unit
        self.xy_unit = xy_unit
        self.coord_array = coord_array

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def plot(self, img_name: str, ref_point: tuple[float, float], cmap_limits: tuple[float, float], cmap_type="hot",
             show=False, draw_circle=False, plot_trajectory=False, trajectory_x_arr=None, trajectory_y_arr=None,
             plot_good_trajectories=False, trajectories_plot_name=None, main_axis=None):
        dim_x = self.shape[0]
        dim_y = self.shape[1]
        x_start = self.x_center - 0.5 * self.xrange
        y_start = self.y_center - 0.5 * self.yrange
        dx = self.xrange / dim_x
        dy = self.yrange / dim_y
        if self.map_type == "coords":
            center_x = (self.x_center - ref_point[0]) * km_in_au
            center_y = (self.y_center - ref_point[1]) * km_in_au
            x_axis = np.array([x_start + _ * dx for _ in range(dim_x)]) * km_in_au - ref_point[0] * km_in_au
            y_axis = np.array([y_start + _ * dy for _ in range(dim_y)]) * km_in_au - ref_point[1] * km_in_au
        elif self.map_type == "velocity":
            center_x = (self.x_center - ref_point[0]) * mps_in_au_per_t_unit
            center_y = (self.y_center - ref_point[1]) * mps_in_au_per_t_unit
            x_axis = np.array([x_start + _ * dx
                               for _ in range(dim_x)]) * mps_in_au_per_t_unit - ref_point[0] * mps_in_au_per_t_unit
            y_axis = np.array([y_start + _ * dy
                               for _ in range(dim_y)]) * mps_in_au_per_t_unit - ref_point[1] * mps_in_au_per_t_unit
        else:
            raise ValueError()
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
        c = ax.pcolormesh(x_axis, y_axis, np.transpose(self.data[:, :, 0] / (2.0 * pi)), cmap=cmap_type,
                          vmin=cmap_limits[0], vmax=cmap_limits[1])
        if self.map_type == "coords":
            ax.set_xlabel(f"$\\Delta x$, {self.xy_unit}")
            ax.set_ylabel(f"$\\Delta y$, {self.xy_unit}")
        elif self.map_type == "velocity":
            ax.set_xlabel(f"$\\Delta v_x$, {self.xy_unit}")
            ax.set_ylabel(f"$\\Delta v_y$, {self.xy_unit}")
        ax.axis([np.min(x_axis), np.max(x_axis), np.min(y_axis), np.max(y_axis)])
        # ax.scatter([(self.x_center - ref_point[0]) * km_in_au], [(self.y_center - ref_point[1]) * km_in_au],
        #            color="yellow", s=1)
        if draw_circle:
            r_circle_sq = center_x * center_x + center_y * center_y
            print(f"Plot circular orbit: R = {sqrt(r_circle_sq)} km")
            ax.plot(x_axis, np.sqrt(r_circle_sq - x_axis * x_axis), color="pink", linewidth=0.75)
            ax.plot(x_axis, -np.sqrt(r_circle_sq - x_axis * x_axis), color="pink", linewidth=0.75)
        if plot_trajectory and (trajectory_x_arr is not None and trajectory_y_arr is not None):
            if self.map_type == "coords":
                ax.plot((trajectory_x_arr - ref_point[0]) * km_in_au, (trajectory_y_arr - ref_point[1]) * km_in_au,
                        color="cyan", linewidth=0.8)
            else:
                raise RuntimeWarning(f"Plot trajectory is not supported for map_type = velocity")
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label(f"$t$, {years_unit}", y=0.5)
        f = plt.gcf()
        f.savefig(img_name, dpi=600)
        if show:
            plt.show()
        plt.cla()
        plt.clf()
        if plot_good_trajectories and trajectories_plot_name is not None and main_axis is not None:
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel(f"$x$, $10^6$ {km_unit}")
            plt.ylabel(f"$y$, $10^6$ {km_unit}")
            plt.xlim([-0.15, 2.1])
            plt.ylim([-0.85, 0.85])
            plt.scatter([0.0], [0.0], color="blue", s=5)
            plt.scatter([(x_l2 - x_earth) * km_in_au * 1.0E-6], [0.0], color="red", s=3)
            max_time_indices = []
            if main_axis == "x":
                for jj in range(dim_x):
                    max_time_indices.append(int(np.argmax(self.data[jj, :, 0])))
                for jjj in range(5, dim_x - 5):
                    median_position = int(np.median(max_time_indices[(jjj - 5):(jjj + 5)]))
                    border1 = max([0, median_position - 6])
                    border2 = min([median_position + 6, dim_y])
                    for jj in range(jjj - 5, jjj + 5):
                        max_time_indices[jj] = border1 + int(np.argmax(self.data[jj, border1:border2, 0]))
                if self.map_type == "coords":
                    plt.plot(np.array([x_start + _ * dx - x_earth for _ in range(dim_x)]) * km_in_au * 1.0E-6,
                             (np.array(max_time_indices) * dy + y_start) * km_in_au * 1.0E-6, color="red",
                             linewidth=0.5)
                elif self.map_type == "velocity":
                    plt.scatter([(self.coord_array.get_body(0).get_coordinate_array("x")[0, 0] - x_earth) * km_in_au * 1.0E-6],
                                [self.coord_array.get_body(0).get_coordinate_array("y")[0, 0] * km_in_au * 1.0E-6],
                                color="red", s=1)
                # print(max_time_indices)
                for jj in range(dim_x):
                    coords = self.coord_array.get_body(jj * dim_y + max_time_indices[jj])
                    x_arr = coords.get_coordinate_array("x")[0:int(self.data[jj, max_time_indices[jj], 2]), 0]
                    y_arr = coords.get_coordinate_array("y")[0:int(self.data[jj, max_time_indices[jj], 2]), 0]
                    plt.plot((x_arr - x_earth) * km_in_au * 1.0E-6, y_arr * km_in_au * 1.0E-6, color="green",
                             linewidth=0.05)
            elif main_axis == "y":
                for jj in range(dim_y):
                    max_time_indices.append(int(np.argmax(self.data[:, jj, 0])))
                for jjj in range(5, dim_y - 5):
                    median_position = int(np.median(max_time_indices[(jjj - 5):(jjj + 5)]))
                    border1 = max([0, median_position - 6])
                    border2 = min([median_position + 6, dim_x])
                    for jj in range(jjj - 5, jjj + 5):
                        max_time_indices[jj] = border1 + int(np.argmax(self.data[border1:border2, jj, 0]))
                if self.map_type == "coords":
                    plt.plot((np.array(max_time_indices) * dx + x_start - x_earth) * km_in_au * 1.0E-6,
                             np.array([y_start + _ * dy for _ in range(dim_y)]) * km_in_au * 1.0E-6,  color="red",
                             linewidth=0.5)
                elif self.map_type == "velocity":
                    plt.scatter([(self.coord_array.get_body(0).get_coordinate_array("x")[0, 0] - x_earth) * km_in_au * 1.0E-6],
                                [self.coord_array.get_body(0).get_coordinate_array("y")[0, 0] * km_in_au * 1.0E-6],
                                color="red", s=1)
                # print(max_time_indices)
                for jj in range(dim_y):
                    coords = self.coord_array.get_body(jj + max_time_indices[jj] * dim_y)
                    x_arr = coords.get_coordinate_array("x")[0:int(self.data[max_time_indices[jj], jj, 2]), 0]
                    y_arr = coords.get_coordinate_array("y")[0:int(self.data[max_time_indices[jj], jj, 2]), 0]
                    plt.plot((x_arr - x_earth) * km_in_au * 1.0E-6, y_arr * km_in_au * 1.0E-6, color="green",
                             linewidth=0.05)
            else:
                raise ValueError(f"Invalid value: main_axis = {main_axis}")
            f = plt.gcf()
            f.savefig(trajectories_plot_name, dpi=600)
            plt.cla()
            plt.clf()


def create_2d_max_stable_time_array(center: tuple[float, float], v_center: tuple[float, float],
                                    xyv_range: tuple[float, float], xyv_dim: tuple[int, int], n_steps: int, m: int,
                                    t_step: float, t0: float, stage_number: int, model: str, label: str,
                                    map_type="coords", xy_unit=UnitName("km", "км")):
    t_unit = t_step * m
    stage_dir = f"stage{stage_number}"
    data_name = f"{data_dir}/{stage_dir}/{label}.dat"
    res_name = f"{result_dir}/{stage_dir}/{label}.fits"
    config_name = f"{config_dir}/{stage_dir}/{label}.config"
    int_params = IntegratorParameters(n_steps, t_step, m, xyv_dim[0] * xyv_dim[1], model, data_name, res_name, t0=t0)
    int_params.save(config_name)
    if integrate:
        if map_type == "coords":
            x_start = center[0] - 0.5 * xyv_range[0]
            y_start = center[1] - 0.5 * xyv_range[1]
            dx = xyv_range[0] / xyv_dim[0]
            dy = xyv_range[1] / xyv_dim[1]
            with open(data_name, "w") as data_out:
                for i in range(xyv_dim[0]):
                    xx = x_start + i * dx
                    for j in range(xyv_dim[1]):
                        yy = y_start + j * dy
                        # line number i * xy_dim[1] + j
                        data_out.write(f"{xx:.20f} {yy:.20f} 0.0 {v_center[0]:.20f} {v_center[1]:.20f} 0.0\n")
        elif map_type == "velocity":
            vx_start = v_center[0] - 0.5 * xyv_range[0]
            vy_start = v_center[1] - 0.5 * xyv_range[1]
            dvx = xyv_range[0] / xyv_dim[0]
            dvy = xyv_range[1] / xyv_dim[1]
            with open(data_name, "w") as data_out:
                for i in range(xyv_dim[0]):
                    vvx = vx_start + i * dvx
                    for j in range(xyv_dim[1]):
                        vvy = vy_start + j * dvy
                        data_out.write(f"{center[0]:.20f} {center[1]:.20f} 0.0 {vvx:.20f} {vvy:.20f} 0.0\n")
        else:
            raise ValueError(f"Invalid value: map_type = {map_type}")
        integrator = Integrator(config_name)
        integrator.start()
        if integrator.exit_code != 0:
            raise RuntimeError("Error in integrator!")
    else:
        integrator = Integrator(config_name)
        integrator.set_result_without_integrating()
    coord_arr = integrator.result
    stable_time = np.zeros((xyv_dim[0], xyv_dim[1], 3))
    for i in range(xyv_dim[0]):
        loc_id0 = i * xyv_dim[1]
        for j in range(xyv_dim[1]):
            x_arr = coord_arr.get_body(loc_id0 + j).get_coordinate_array("x")[:, 0]
            is_stable = False
            for tt in range(len(x_arr)):
                if (not is_stable) and lim2 > x_arr[tt] > lim1:
                    is_stable = True
                    stable_time[i, j, 1] = tt
                    continue
                if is_stable and (x_arr[tt] < lim1 or x_arr[tt] > lim2):
                    stable_time[i, j, 2] = tt
                    stable_time[i, j, 0] = (stable_time[i, j, 2] - stable_time[i, j, 1]) * t_unit
                    break
    if map_type == "coords":
        return Map2D(stable_time, center, xyv_range, t_unit, coord_array=coord_arr)
    elif map_type == "velocity":
        return Map2D(stable_time, v_center, xyv_range, t_unit, map_type=map_type, xy_unit=xy_unit,
                     coord_array=coord_arr)
    else:
        raise RuntimeError("Non-reachable error. How did you make it?")


def stage5(model="restricted_3body", stage_number=5):
    print(f"\n***** STAGE {stage_number} *****\n")
    stage_dir = f"stage{stage_number}"
    os.makedirs(f"{data_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{config_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{result_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{log_dir}/{stage_dir}", exist_ok=True)
    os.makedirs(f"{img_dir}/{stage_dir}", exist_ok=True)
    cmap_limits = (0.0, 1.0)
    params = read_config(f"{log_dir}/stage{stage_number - 2}/stage{stage_number - 2}.log")
    t_limit = int(params["t_limit"])
    t_unit_1 = float(params["t_unit_1"])
    t_unit_2 = float(params["t_unit_2"])
    min_earth_dist_id = int(params["min_earth_dist_id"])
    # max_earth_dist_id = int(params["max_earth_dist_id"])
    interesting1_id = int(params["interesting1_id"])
    # coord1 = CartesianCoordinateArray.from_file(f"{result_dir}/stage{stage_number - 2}/trajectories.fits", "FITS")
    coord2 = CartesianCoordinateArray.from_file(f"{result_dir}/stage{stage_number - 2}/earth_trajectories.fits", "FITS")
    # minimum Earth distance orbit
    if model == "restricted_3body":
        min_earth_dist_x_arr1 = coord2.get_body(min_earth_dist_id).get_coordinate_array("x")[:, 0]
        min_earth_dist_y_arr1 = coord2.get_body(min_earth_dist_id).get_coordinate_array("y")[:, 0]
        min_earth_dist_vx_arr1 = coord2.get_body(min_earth_dist_id).get_coordinate_array("vx")[:, 0]
        min_earth_dist_vy_arr1 = coord2.get_body(min_earth_dist_id).get_coordinate_array("vy")[:, 0]
        earth_r_x = min_earth_dist_x_arr1 - x_earth
        earth_r_y = min_earth_dist_y_arr1
        earth_v_rel_x = min_earth_dist_vx_arr1
        earth_v_rel_y = min_earth_dist_vy_arr1
        earth_dist_sq_arr2 = earth_r_x * earth_r_x + earth_r_y * earth_r_y
        r_cdot_v_arr = earth_r_x * earth_v_rel_x + earth_r_y * earth_v_rel_y
        ref_circle_dist = 100000 / km_in_au
        ref_circle_dist_sq = ref_circle_dist * ref_circle_dist
        min_earth_dist_ref_t0 = 0
        for i in range(1, len(r_cdot_v_arr)):
            if r_cdot_v_arr[i] * r_cdot_v_arr[i - 1] <= 0.0 and earth_dist_sq_arr2[i] < ref_circle_dist_sq:
                min_earth_dist_ref_t0 = i
                break
        xy_range = (10000 / km_in_au, 10000 / km_in_au)
        xy_dim = (200, 200)
        v_range = (100 / mps_in_au_per_t_unit, 100 / mps_in_au_per_t_unit)
        v_dim = (200, 200)
        m1 = 10
        t_step = t_unit_1 / m1
        t_unit_1 = t_step * m1
        min_earth_dist_t0 = -t_limit * t_unit_1 - min_earth_dist_ref_t0 * t_unit_2
        # n_steps = ceil(-min_earth_dist_t0 / t_step) * 3
        n_steps = ceil(6.5E7 / s_in_time_unit / t_step)
        min_earth_dist_ref_x = min_earth_dist_x_arr1[min_earth_dist_ref_t0]
        min_earth_dist_ref_y = min_earth_dist_y_arr1[min_earth_dist_ref_t0]
        min_earth_dist_ref_vx = min_earth_dist_vx_arr1[min_earth_dist_ref_t0]
        min_earth_dist_ref_vy = min_earth_dist_vy_arr1[min_earth_dist_ref_t0]
        min_earth_dist_ref_coords = (min_earth_dist_ref_x, min_earth_dist_ref_y)
        min_earth_dist_ref_v = (min_earth_dist_ref_vx, min_earth_dist_ref_vy)
        if not plot_velocity:
            stable_time_arr = create_2d_max_stable_time_array(min_earth_dist_ref_coords, min_earth_dist_ref_v, xy_range,
                                                              xy_dim, n_steps, m1, t_step, min_earth_dist_t0,
                                                              stage_number, model, "min_earth_dist_trajectories")
            plot_x_arr = min_earth_dist_x_arr1[(min_earth_dist_ref_t0 - 100):(min_earth_dist_ref_t0 + 100)]
            plot_y_arr = min_earth_dist_y_arr1[(min_earth_dist_ref_t0 - 100):(min_earth_dist_ref_t0 + 100)]
            stable_time_arr.plot(f"{img_dir}/{stage_dir}/min_earth_dist_trajectories.png", (x_earth, 0.0), cmap_limits,
                                 plot_trajectory=True, trajectory_x_arr=plot_x_arr, trajectory_y_arr=plot_y_arr,
                                 plot_good_trajectories=True, draw_circle=True,
                                 trajectories_plot_name=f"{img_dir}/{stage_dir}/min_dist_all_trajectories.png",
                                 main_axis="x")
        if plot_velocity:
            stable_time_arr = create_2d_max_stable_time_array(min_earth_dist_ref_coords, min_earth_dist_ref_v, v_range,
                                                              v_dim, n_steps, m1, t_step, min_earth_dist_t0, stage_number,
                                                              model, "min_earth_dist_vel_trajectories",
                                                              map_type="velocity", xy_unit=mps_unit)
            stable_time_arr.plot(f"{img_dir}/{stage_dir}/min_earth_dist_vel_trajectories.png", min_earth_dist_ref_v,
                                 cmap_limits, plot_good_trajectories=True,
                                 trajectories_plot_name=f"{img_dir}/{stage_dir}/min_dist_all_trajectories_v.png",
                                 main_axis="y")
    # interesting orbit
    interesting1_x_arr1 = coord2.get_body(interesting1_id).get_coordinate_array("x")[:, 0]
    interesting1_y_arr1 = coord2.get_body(interesting1_id).get_coordinate_array("y")[:, 0]
    interesting1_vx_arr1 = coord2.get_body(interesting1_id).get_coordinate_array("vx")[:, 0]
    interesting1_vy_arr1 = coord2.get_body(interesting1_id).get_coordinate_array("vy")[:, 0]
    if model == "bicircular_r4bp":
        t_ax = np.array([-t_limit * t_unit_1 - _ * t_unit_2 for _ in range(len(interesting1_x_arr1))])
        x_earth_arr, y_earth_arr = earth_coords(t_ax)
        vx_earth_arr, vy_earth_arr = earth_velocity(t_ax)
        earth_r_x = interesting1_x_arr1 - x_earth_arr
        earth_r_y = interesting1_y_arr1 - y_earth_arr
        earth_v_rel_x = interesting1_vx_arr1 - vx_earth_arr
        earth_v_rel_y = interesting1_vy_arr1 - vy_earth_arr
    else:
        earth_r_x = interesting1_x_arr1 - x_earth
        earth_r_y = interesting1_y_arr1
        earth_v_rel_x = interesting1_vx_arr1
        earth_v_rel_y = interesting1_vy_arr1
    earth_dist_sq_arr2 = earth_r_x * earth_r_x + earth_r_y * earth_r_y
    r_cdot_v_arr = earth_r_x * earth_v_rel_x + earth_r_y * earth_v_rel_y
    ref_circle_dist = 300000 / km_in_au
    ref_circle_dist_sq = ref_circle_dist * ref_circle_dist
    interesting1_ref_t0 = 0
    for i in range(1, len(r_cdot_v_arr)):
        if r_cdot_v_arr[i] * r_cdot_v_arr[i - 1] <= 0.0 and earth_dist_sq_arr2[i] < ref_circle_dist_sq:
            interesting1_ref_t0 = i
            break
    interesting1_ref_x = interesting1_x_arr1[interesting1_ref_t0]
    interesting1_ref_y = interesting1_y_arr1[interesting1_ref_t0]
    interesting1_ref_vx = interesting1_vx_arr1[interesting1_ref_t0]
    interesting1_ref_vy = interesting1_vy_arr1[interesting1_ref_t0]
    xy_range = (10000 / km_in_au, 10000 / km_in_au)
    xy_dim = (200, 200)
    v_range = (100 / mps_in_au_per_t_unit, 100 / mps_in_au_per_t_unit)
    v_dim = (200, 200)
    interesting1_t0 = -t_limit * t_unit_1 - interesting1_ref_t0 * t_unit_2
    m1 = 10
    t_step = 2.0 * t_unit_1 / m1
    # t_unit_1 = t_step * m1
    n_steps = ceil(-interesting1_t0 / t_step) * 3
    interesting1_ref_coords = (interesting1_ref_x, interesting1_ref_y)
    interesting1_ref_v = (interesting1_ref_vx, interesting1_ref_vy)
    if not plot_velocity:
        stable_time_arr = create_2d_max_stable_time_array(interesting1_ref_coords, interesting1_ref_v, xy_range, xy_dim,
                                                          n_steps, m1, t_step, interesting1_t0, stage_number, model,
                                                          "interesting_trajectories")
        if model == "bicircular_r4bp":
            t0_x_earth, t0_y_earth = earth_coords(interesting1_t0)
        else:
            t0_x_earth = x_earth
            t0_y_earth = 0.0
        plot_x_arr = interesting1_x_arr1[(interesting1_ref_t0 - 100):(interesting1_ref_t0 + 100)]
        plot_y_arr = interesting1_y_arr1[(interesting1_ref_t0 - 100):(interesting1_ref_t0 + 100)]
        stable_time_arr.plot(f"{img_dir}/{stage_dir}/interesting_trajectories.png", (t0_x_earth, t0_y_earth), cmap_limits,
                             draw_circle=True, plot_trajectory=True, trajectory_x_arr=plot_x_arr,
                             trajectory_y_arr=plot_y_arr, plot_good_trajectories=True,
                             trajectories_plot_name=f"{img_dir}/{stage_dir}/interesting_all_trajectories.png",
                             main_axis="x")
    if plot_velocity:
        stable_time_arr = create_2d_max_stable_time_array(interesting1_ref_coords, interesting1_ref_v, v_range, v_dim,
                                                          n_steps, m1, t_step, interesting1_t0, stage_number, model,
                                                          "interesting_trajectories_v", map_type="velocity",
                                                          xy_unit=mps_unit)
        stable_time_arr.plot(f"{img_dir}/{stage_dir}/interesting_trajectories_v.png", interesting1_ref_v, cmap_limits,
                             plot_good_trajectories=True,
                             trajectories_plot_name=f"{img_dir}/{stage_dir}/interesting_all_trajectories_v.png",
                             main_axis="y")


def stage6():
    stage5(model="bicircular_r4bp", stage_number=6)


plt.rcParams["figure.constrained_layout.use"] = True
km_in_au = 149597870.7
s_in_time_unit = 31558149.8 / (2 * pi)
print(f"1 t_unit = {s_in_time_unit} s = {1 / (2 * pi)} years")
mps_in_au_per_t_unit = 1000 * km_in_au / s_in_time_unit
print(f"1 v_unit = {mps_in_au_per_t_unit} m/s")

data_dir = "result/data"
config_dir = "result/data/config"
result_dir = "result/result"
log_dir = "result/log"
img_dir = "result/img"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stage", action="store", default=0)
parser.add_argument("-i", "--integrate", action="store", default="True")
parser.add_argument("--russian", action="store_true")
parser.add_argument("-v", "--plot_velocity", action="store_true")
args = parser.parse_args()
stages = args.stage
if args.integrate.lower() in ["false", "no", "n"]:
    integrate = False
elif args.integrate.lower() in ["true", "yes", "y"]:
    integrate = True
else:
    raise ValueError(f"Invalid word {args.integrate}")
if stages == "all":
    stages = 123456
stages_list = [int(_) for _ in list(str(stages))]
if args.stage == 0:
    print("Nothing to do")
russian = args.russian
plot_velocity = args.plot_velocity


def eq_for_libration_points(x):
    return x - mu2 / (x * fabs(x)) - (1.0 - mu2) / ((1 + x) * fabs(1 + x)) + 1.0 - mu2


mu2 = (398600.435507 + 4902.800118) / 132712440041.279419
print(f"mu_2 = {mu2}")
l2_loc = (mu2 / 3.0 / (1.0 - mu2)) ** (1.0 / 3.0)
l1_loc = -l2_loc
l3_loc = -2.0 + 7.0 / 12.0 * mu2
l2_loc = optimize.fsolve(eq_for_libration_points, l2_loc)[0]
l1_loc = optimize.fsolve(eq_for_libration_points, l1_loc)[0]
l3_loc = optimize.fsolve(eq_for_libration_points, l3_loc)[0]
aa = mu2 / (l2_loc * l2_loc * l2_loc) + (1 - mu2) / ((1 + l2_loc) * (1 + l2_loc) * (1 + l2_loc))
print(f"a = {aa}")
lambda_coeff = sqrt(0.5 * (sqrt(9 * aa * aa - 8 * aa) + aa - 2))
omega_coeff = sqrt(0.5 * (sqrt(9 * aa * aa - 8 * aa) - aa + 2))
print(f"lambda = {lambda_coeff}; omega = {omega_coeff}")
k1 = (lambda_coeff * lambda_coeff - 2 * aa - 1) / (2 * lambda_coeff)
k2 = -(omega_coeff * omega_coeff + 2 * aa + 1) / (2 * omega_coeff)
print(f"k1 = {k1}; k2 = {k2}")
delta_x0 = -277548.0 / km_in_au
print(f"delta x_0 = {delta_x0 * km_in_au} km")
x0 = 1.0 - mu2 + l2_loc + delta_x0
x_l2 = 1.0 - mu2 + l2_loc
x_l1 = 1.0 - mu2 + l1_loc
x_l3 = 1.0 - mu2 + l3_loc
print(f"L1: x = {x_l1}; delta_1 = {l1_loc}")
print(f"L2: x = {x_l2}; delta_2 = {l2_loc}")
print(f"L3: x = {x_l3}; delta_3 = {l3_loc}")
print(f"x_0 = {x_l2 + delta_x0} AU")
vy0_init = k2 * omega_coeff * delta_x0
print(f"Initial approximation: vy_0 = {vy0_init * mps_in_au_per_t_unit}")
x_earth = 1.0 - mu2
lim1 = x_l2 - 500000 / km_in_au
lim2 = x_l2 + 500000 / km_in_au
n_l = 13.36874661478304
moon_period = 2.0 * pi / n_l
delta = 0.012150584394709708
ll = 0.00257
r_earth = 6378.0 / km_in_au
r_earth_sq = r_earth * r_earth

km_unit = UnitName("km", "км")
mps_unit = UnitName("m / s", "м / с")
au_unit = UnitName("AU", "а. е.")
years_unit = UnitName("years", "лет")

global_params = read_config("global_parameters.config")
if "log2S" in global_params.keys():
    global_collo_log2s = int(global_params["log2S"])
else:
    global_collo_log2s = 3
global_platform = global_params["platform"].strip('"')
global_device = global_params["device"].strip('"')
print(f"platform = {global_platform}; device = {global_device}")

stages = [stage1, stage2, stage3, stage4, stage5, stage6]
for stage_num in range(len(stages)):
    if stage_num + 1 in stages_list:
        stages[stage_num]()
