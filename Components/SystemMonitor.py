import threading
import time
import psutil

class SystemMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.active = False
        self.thread = None
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_handle = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml
        except (ImportError, Exception):
            self.pynvml = None
            print("[Monitor] pynvml no disponible. No se monitoreará la GPU.")

    def _monitor_loop(self):
        """El loop que se ejecuta en el hilo para recolectar datos."""
        while self.active:
            # CPU Usage
            self.cpu_usage.append(psutil.cpu_percent())

            # GPU Usage
            if self.pynvml and self.gpu_handle:
                try:
                    gpu_util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    self.gpu_usage.append(gpu_util.gpu)
                except self.pynvml.NVMLError:
                    self.gpu_usage.append(0)
            
            time.sleep(self.interval)

    def start(self):
        """Inicia el hilo de monitoreo."""
        if self.thread is not None and self.thread.is_alive():
            return

        self.cpu_usage = []
        self.gpu_usage = []
        self.active = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("[Monitor] Iniciando monitoreo de sistema...")

    def stop(self) -> str:
        """Detiene el monitoreo y devuelve un string con las estadísticas."""
        if not self.active or self.thread is None:
            return "[Monitor] El monitoreo no estaba activo."
        
        self.active = False
        self.thread.join()
        print("[Monitor] Monitoreo detenido.")

        if not self.cpu_usage:
            return "[Monitor] No se recolectaron datos."

        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
        max_cpu = max(self.cpu_usage)
        
        stats_str = f"[bold]CPU Usage:[/bold] Avg: {avg_cpu:.1f}% | Max: {max_cpu:.1f}%"

        if self.gpu_usage:
            avg_gpu = sum(self.gpu_usage) / len(self.gpu_usage)
            max_gpu = max(self.gpu_usage)
            stats_str += f"  [bold]GPU Usage:[/bold] Avg: {avg_gpu:.1f}% | Max: {max_gpu:.1f}%"
        
        return stats_str

    def __del__(self):
        if self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
