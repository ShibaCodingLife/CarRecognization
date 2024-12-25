package com.neusoft.core.util;

import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;

import org.apache.velocity.VelocityContext;
import org.apache.velocity.app.Velocity;

/**
 * @ClassName: VelocityUtil
 * @Description: 模板解析工具类
 */
public class VelocityUtil {

	/**
	 * @Title: parseTemplate
	 * @Description: 解析模板方法
	 * @param template
	 *            模板内容字符串
	 * @param params
	 *            用于解析的变量，与模板中的变量一一对应
	 * @return String 返回解析完成的内容
	 */
	public static String parseTemplate(String template,
			Map<String, Object> params) {
		VelocityContext context = new VelocityContext(
				new HashMap<String, Object>(params));
		StringWriter w = new StringWriter();
		Velocity.evaluate(context, w, VelocityUtil.class.getName(), template);
		return w.toString();
	}

	/**
	 * @Title: main
	 * @Description: 使用方法
	 * @param args
	 */
	public static void main(String[] args) {
		String temp = "hello $username,Welcome to DL";
		Map<String, Object> params = new HashMap<String, Object>();
		params.put("username", "张三");
		String content = parseTemplate(temp, params);
		System.out.println(content);
	}
}
