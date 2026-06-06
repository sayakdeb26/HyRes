#!/usr/bin/env python3
import time
import os
import csv
import psutil

# Try importing pynvml
try:
    import pynvml
    pynvml.nvmlInit()
    nvml_available = True
except Exception:
    nvml_available = False

def get_gpu_stats():
    if not nvml_available:
        # Fallback to parsing nvidia-smi output
        try:
            import subprocess
            out = subprocess.check_output([
                "nvidia-smi", 
                "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu", 
                "--format=csv,noheader,nounits"
            ]).decode('utf-8').strip()
            parts = [float(p.strip()) for p in out.split(',')]
            return {
                'gpu_util': parts[0],
                'vram_used': parts[1],
                'gpu_power': parts[2],
                'gpu_temp': parts[3]
            }
        except Exception:
            return {'gpu_util': 0.0, 'vram_used': 0.0, 'gpu_power': 0.0, 'gpu_temp': 0.0}
            
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # milliwatts to watts
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            'gpu_util': float(util.gpu),
            'vram_used': float(mem.used) / (1024.0 * 1024.0), # to MB
            'gpu_power': float(power),
            'gpu_temp': float(temp)
        }
    except Exception:
        return {'gpu_util': 0.0, 'vram_used': 0.0, 'gpu_power': 0.0, 'gpu_temp': 0.0}

def get_cpu_ram():
    cpu_util = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024.0 * 1024.0) # to MB
    return cpu_util, ram_used

def main():
    os.makedirs('/home/sayak/HyRes/experiment_results/resource_usage', exist_ok=True)
    log_path = '/home/sayak/HyRes/experiment_results/resource_usage/resource_log.csv'
    header = ["timestamp", "cpu_utilization", "ram_used_mb", "gpu_utilization", "vram_used_mb", "gpu_power_watts", "gpu_temp_c"]
    
    file_exists = os.path.exists(log_path)
    
    with open(log_path, 'a') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
            
        print("Starting resource monitor...")
        try:
            # Initialize CPU percent call to establish baseline
            psutil.cpu_percent(interval=None)
            time.sleep(0.1)
            while True:
                ts = time.time()
                cpu_util, ram_used = get_cpu_ram()
                gpu = get_gpu_stats()
                writer.writerow([
                    ts,
                    cpu_util,
                    ram_used,
                    gpu['gpu_util'],
                    gpu['vram_used'],
                    gpu['gpu_power'],
                    gpu['gpu_temp']
                ])
                f.flush()
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Resource monitor stopped.")

if __name__ == '__main__':
    main()
