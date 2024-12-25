package com.neusoft.neumooc.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

import com.neusoft.core.common.BaseController;

/**
 * @ClassName: CommonController
 * @Description: 公共控制类
 */
@Controller
@RequestMapping(value = "/common")
public class CommonController extends BaseController {

	/**
	 * @Title: getNotFoundPage
	 * @Description: 进入404页面
	 * @return 404页面
	 */
	@RequestMapping(value = "/404")
	public String getNotFoundPage() {
		return "/common/404";
	}

	/**
	 * @Title: getInternalError
	 * @Description: 进入500页面
	 * @return 500页面
	 */
	@RequestMapping(value = "/500")
	public String getInternalError() {
		return "/common/500";
	}

}
