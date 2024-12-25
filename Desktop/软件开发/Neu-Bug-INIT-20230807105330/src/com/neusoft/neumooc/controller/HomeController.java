package com.neusoft.neumooc.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

import com.neusoft.core.common.BaseController;

/**
 * @ClassName: HomeController
 * @Description: 首页控制类
 */
@Controller
@RequestMapping(value = "/home")
public class HomeController extends BaseController {

	/**
	 * @Title: index
	 * @Description: 进入首页
	 * @return 首页
	 */
	@RequestMapping(value = "/index")
	public String index() {
		return "home/index";
	}
}
