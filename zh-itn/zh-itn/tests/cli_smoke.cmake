execute_process(
  COMMAND "${ZH_ITN_EXE}"
  INPUT_FILE "${ZH_ITN_REQUEST}"
  OUTPUT_VARIABLE response
  ERROR_VARIABLE diagnostic
  RESULT_VARIABLE status
)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "zh_itn failed: ${status}; ${diagnostic}; ${response}")
endif()
if(NOT response MATCHES "\"output\":\"在2026年发布\"")
  message(FATAL_ERROR "zh_itn returned unexpected output: ${response}")
endif()
