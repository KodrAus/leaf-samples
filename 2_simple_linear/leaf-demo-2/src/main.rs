extern crate leaf;
extern crate collenchyma as co;

use std::sync::{ Arc, RwLock };
use std::rc::Rc;
use leaf::layer::*;
use leaf::layers::*;
use co::prelude::*;

fn main() {
    let gpu = Rc::new(Backend::<Cuda>::default().unwrap());
    let mut net_config = SequentialConfig::default();

    let data = vec![1, 2, 3];
    let data_len = data.len();

    net_config.add_input("data", &data);

    net_config.add_layer(LayerConfig::new("linear", 
    	LinearConfig {
    		output_size: data_len
    	}
    ));

    let mut net = Layer::from_config(
    	gpu.clone(), 
    	&LayerConfig::new(
    		"network", 
    		LayerType::Sequential(net_config)
    	)
    );

    let gpu_data = SharedTensor::<f32>::new(
		gpu.device(), 
		&data
	).unwrap();

    let gpu_lock = Arc::new(RwLock::new(gpu_data));

    net.forward(&[gpu_lock.clone()]);
}
