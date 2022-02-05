
"""
========================================      MY DEVICE     ========================================
======================================== System Information ========================================
System: Windows
Node Name: WIN-JU7ILL7KI98
Release: 10
Version: 10.0.19041
Machine: AMD64
Processor: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel 
======================================== CPU Info ========================================
Physical cores: 6
Total cores: 12
Max Frequency: 2208.00Mhz
Min Frequency: 0.00Mhz
Current Frequency: 2208.00Mhz
CPU Usage Per Core:
Core 0: 22.2%
Core 1: 0.0%
Core 2: 1.5%
Core 3: 3.0%
Core 4: 7.7%
Core 5: 0.0%
Core 6: 3.1%
Core 7: 0.0%
Core 8: 0.0%
Core 9: 0.0%
Core 10: 4.6%
Core 11: 0.0%
Total CPU Usage: 4.0%
======================================== Disk Information ========================================
Partitions and Usage:
=== Device: C:\ ===
  Mountpoint: C:\
  File system type: NTFS
  Total Size: 237.42GB
  Used: 113.23GB
  Free: 124.19GB
  Percentage: 47.7%
=== Device: D:\ ===
  Mountpoint: D:\
  File system type: NTFS
  Total Size: 871.51GB
  Used: 491.02GB
  Free: 380.49GB
  Percentage: 56.3%
=== Device: E:\ ===
  Mountpoint: E:\
  File system type: CDFS
  Total Size: 4.14GB
  Used: 4.14GB
  Free: 0.00B
  Percentage: 100.0%
Total read: 45.68GB
Total write: 49.55GB
======================================== Memory Information ========================================
Total: 7.85GB
Available: 1.80GB
Used: 6.05GB
Percentage: 77.0%
==================== SWAP ====================
Total: 16.35GB
Free: 5.18GB
Used: 11.18GB
Percentage: 68.3%
======================================== GPU Details ========================================
  id  name                 load    free memory    used memory    total memory    temperature    uuid
----  -------------------  ------  -------------  -------------  --------------  -------------  ----------------------------------------
   0  GeForce GTX 1050 Ti  11.0%   3243.0MB       853.0MB        4096.0MB        53.0 °C        GPU-89d4340b-b1b4-9191-7414-2c1003d17f1f
"""
import psutil
import platform
from datetime import datetime
import GPUtil
from tabulate import tabulate



def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def Print_System_Info():
    # System Info
    print("="*40, "System Information", "="*40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")

def Print_Boot_Time():
    # Boot Time
    print("="*40, "Boot Time", "="*40)
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    print(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

def Print_CPU_Info():   
    # let's print CPU information
    print("="*40, "CPU Info", "="*40)
    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    # CPU usage
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")

def Print_Memory_Info():
    # Memory Info
    print("="*40, "Memory Information", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")
    print("="*20, "SWAP", "="*20)
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    print(f"Total: {get_size(swap.total)}")
    print(f"Free: {get_size(swap.free)}")
    print(f"Used: {get_size(swap.used)}")
    print(f"Percentage: {swap.percent}%")

def Print_Disk_Info():
    # Disk Information
    print("="*40, "Disk Information", "="*40)
    print("Partitions and Usage:")
    # get all disk partitions
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"=== Device: {partition.device} ===")
        print(f"  Mountpoint: {partition.mountpoint}")
        print(f"  File system type: {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            # this can be catched due to the disk that
            # isn't ready
            continue
        print(f"  Total Size: {get_size(partition_usage.total)}")
        print(f"  Used: {get_size(partition_usage.used)}")
        print(f"  Free: {get_size(partition_usage.free)}")
        print(f"  Percentage: {partition_usage.percent}%")
    # get IO statistics since boot
    disk_io = psutil.disk_io_counters()
    print(f"Total read: {get_size(disk_io.read_bytes)}")
    print(f"Total write: {get_size(disk_io.write_bytes)}")

def Print_Network_Info():
    # Network information
    print("="*40, "Network Information", "="*40)
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            print(f"=== Interface: {interface_name} ===")
            if str(address.family) == 'AddressFamily.AF_INET':
                print(f"  IP Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == 'AddressFamily.AF_PACKET':
                print(f"  MAC Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    print(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")

def Print_GPU_Info():
    # GPU information
    print("="*40, "GPU Details", "="*40)
    gpus = GPUtil.getGPUs()
    list_gpus = []
    for gpu in gpus:
        # get the GPU id
        gpu_id = gpu.id
        # name of GPU
        gpu_name = gpu.name
        # get % percentage of GPU usage of that GPU
        gpu_load = f"{gpu.load*100}%"
        # get free memory in MB format
        gpu_free_memory = f"{gpu.memoryFree}MB"
        # get used memory
        gpu_used_memory = f"{gpu.memoryUsed}MB"
        # get total memory
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        # get GPU temperature in Celsius
        gpu_temperature = f"{gpu.temperature} °C"
        gpu_uuid = gpu.uuid
        list_gpus.append((
            gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
            gpu_total_memory, gpu_temperature, gpu_uuid
        ))
    
    print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory",
                                    "temperature", "uuid")))

def main():
    Print_System_Info()
    Print_CPU_Info()
    Print_Disk_Info()
    Print_Memory_Info()
    Print_GPU_Info()

if __name__ == '__main__':
    main()