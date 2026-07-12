include("src/cc_gpu.jl"); include("src/cc_gpu_wheel.jl")
function t1(k,lo,width,mp)
  st=gpu_wheel_setup(k;max_prime=mp,verbose=false)
  gpu_wheel_scan!(st,lo,lo+width÷20)
  t=@elapsed r=gpu_wheel_scan!(st,lo,lo+width)
  CUDA.unsafe_free!(st.d_out); (t,length(r))
end
for k in (15,16)
  println("### CC$k")
  for mp in (20000,30000,45000)
    (t,f)=t1(k,Int64(10)^16,Int64(2)*10^15,mp)
    println("mp=",rpad(mp,7)," scan=",round(t,digits=3),"s found=",f)
  end
end
println("DONE")
