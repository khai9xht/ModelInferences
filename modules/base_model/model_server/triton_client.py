import tritonclient.grpc as grpcclient
import requests
from modules.configs import ModelServerConfig

class TritonClient:
	def __init__(self, config: ModelServerConfig) -> None:
		triton_ip = config.host
		triton_port = config.port
		url_check = f"{triton_ip}:{triton_port - 1}"
		try:
			res = requests.get("http://" + url_check + "/v2/health/ready")
			assert res.ok and res.status_code == 200, f"[{self.__class__.__name__}]: Failed to connect to model server"    
		except requests.exceptions.RequestException as e:
			print(f"[{self.__class__.__name__}]: {e}")
			
		url = f"{triton_ip}:{triton_port}"
		self.triton_client = grpcclient.InferenceServerClient(url=url)
		self.list_model_configs = {}
		

	def initialize(self, model_name, input_names, input_sizes, output_names, float_modes=None) -> None:
		try:
			assert self.triton_client.is_model_ready(model_name), \
				f"[{self.__class__.__name__}]: Model {model_name} is not ready"
			if float_modes is None:
				float_modes=["FP32"]*len(input_sizes)
			grpc_inputs = [
				grpcclient.InferInput(name, input_size, float_mode) \
				for name, input_size, float_mode, in \
					zip(input_names, input_sizes, float_modes)
			]
			grpc_outputs = [grpcclient.InferRequestedOutput(output) for output in output_names]
			self.list_model_configs[model_name] = {
				"inputs": grpc_inputs,
				"outputs": grpc_outputs
			}
		except:
			print(f"Failed to connect to {model_name} model server.")

	def infer(self, model_name, inputs) -> grpcclient.InferResult:
		try:
			for i in range(len(self.list_model_configs[model_name]["inputs"])):
				self.list_model_configs[model_name]["inputs"][i].set_data_from_numpy(inputs[i])
			results = self.triton_client.infer(
				model_name=model_name,
				inputs=self.list_model_configs[model_name]["inputs"],
				outputs=self.list_model_configs[model_name]["outputs"],
				headers={}
			)
			return results
		except:
			print(f"Failed to request to {model_name} of model server.")
