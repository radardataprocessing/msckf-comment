/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

/**
 * 1. handle make it possible to assign namespace via constructor function
 * 
 * ros::NodeHandle nh("my_namespace");
 * all relative names using the handle is relative to <node_namespace>/my_namespace instead of <nodenamespace>
 * 
 * you can also assign a parent handle and appended namespace
 * ros::NodeHandle nh1("ns1");
 * ros::NodeHandle nh2 (nh1, "ns2");
 * these two statements put nh2 to namespace <node_namespace>/ns1/ns2
 * 
 * 
 * 
 * 2. you can also assign global namespace
 * 
 * ros::NodeHandle nh("/my_global_namespace")
 * this is not recommanded, you can't put node to other namespace in this way
 * 
 * 
 * 
 * 3. private name
 * ros::NodeHandle nh("~my_private_namespace");
 * ros::Subscriber sub = nh.subscribe("my_private_topic",...);
 * will subscribe <node_name>/my_privete_namespace/my_private_topic
 * 
 * 
 * node_name = node_namespace + nodename
 */

#include <msckf_vio/msckf_vio_nodelet.h>

namespace msckf_vio {
void MsckfVioNodelet::onInit() {
  msckf_vio_ptr.reset(new MsckfVio(getPrivateNodeHandle()));
  if (!msckf_vio_ptr->initialize()) {
    ROS_ERROR("Cannot initialize MSCKF VIO...");
    return;
  }
  return;
}

PLUGINLIB_EXPORT_CLASS(msckf_vio::MsckfVioNodelet,
    nodelet::Nodelet);

} // end namespace msckf_vio

