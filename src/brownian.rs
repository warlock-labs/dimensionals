use num_traits::Num;
use rand_distr::{Normal, Distribution};
use crate::{storage::DimensionalStorage, Dimensional};
use std::ops::AddAssign;

// TODO: make it work for all types (now works only for floating types)
// TODO: have the option to save the path of the walk (although may be costly)


fn brownian_motion<T, S, const N: usize>(dimensional: &mut Dimensional<T, S, N>, std_dev: f64, n_steps: usize) 
where
    T: Num + Copy + AddAssign ,
    f64: Into<T>,
    S: DimensionalStorage<T, N>,
{
    let normal = Normal::new(0.0, std_dev).unwrap();
    let length_dimensional = dimensional.len();
    let mut iter = dimensional.iter_mut();
    for _ in 1..=length_dimensional {
        if let Some(elem) = iter.next() {
            for _ in 0..n_steps{
                *elem += normal.sample(&mut rand::thread_rng()).into();
            }
        }
    }

}

mod tests {
    use crate::{brownian::brownian_motion, Dimensional, LinearArrayStorage};
    

    #[test]
    fn test_len() {
        let mut zeros: Dimensional<f64, LinearArrayStorage<f64, 2>, 2> = Dimensional::zeros([2, 3]);
        brownian_motion(&mut zeros,0.1, 1000);  
        println!("{}", format!("{}", zeros));
    
    }

}
