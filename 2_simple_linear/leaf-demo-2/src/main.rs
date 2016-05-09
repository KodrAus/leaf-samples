extern crate leaf;
extern crate collenchyma as co;

use std::sync::{ Arc, RwLock };
use std::rc::Rc;
use leaf::layer::*;
use leaf::layers::*;
use co::prelude::*;

fn main() {
	let native = Backend::<Native>::default().unwrap();
	let gpu = Rc::new(Backend::<Cuda>::default().unwrap());
	let mut net_config = SequentialConfig::default();

	let data = vec![1, 2, 3];

	net_config.add_input("data", &vec![1, 3]);

	net_config.add_layer(LayerConfig::new("linear", 
		LinearConfig {
			output_size: 1
		}
	));

	let mut net = Layer::from_config(
		gpu.clone(), 
		&LayerConfig::new(
			"network", 
			LayerType::Sequential(net_config)
		)
	);

	let mut shared_data = SharedTensor::<f32>::new(
		gpu.device(), 
		&data
	).unwrap();
	shared_data.add_device(native.device()).unwrap();
	let data_lock = Arc::new(RwLock::new(shared_data));

	net.forward(&[data_lock]);
}
