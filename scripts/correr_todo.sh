for t in test_ik test_traj test_jac test_dyn test_planner; do
  echo "################ $t ################"
  python3 $t.py
done
