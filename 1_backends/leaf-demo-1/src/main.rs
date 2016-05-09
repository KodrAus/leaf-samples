extern crate collenchyma as co;

use co::prelude::*;

fn main() {
	let gpu = Backend::<Cuda>::default().unwrap();
	println!("{:?}", gpu.hardwares());

	let cpu = Backend::<Native>::default().unwrap();
	println!("{:?}", cpu.hardwares());
}
