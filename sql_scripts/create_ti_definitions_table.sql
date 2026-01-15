CREATE TABLE ti_definitions (
  ti_def_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  ti_name VARCHAR(24) NOT NULL,
  time_frame INT NOT NULL,
  params_json JSON NOT NULL,
  PRIMARY KEY (ti_def_id),
  UNIQUE KEY uq_ti_definitions_name_time_frame (ti_name, time_frame),
  CHECK (time_frame IN (3, 10, 20))
);
    
    
    