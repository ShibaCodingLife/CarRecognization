package com.neusoft.core.util;

import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @ClassName: SensitiveWordFilter
 * @Description: 敏感词过滤
 */
public class SensitiveWordFilter {

	/**
	 * @Title: execFilter
	 * @Description: 执行过滤器
	 * @param inputWords
	 * @return 过滤后的字符串
	 */
	public static String execFilter(String inputWords) {
		String sensitiveWord = PCache.getSensitiveWord();
		Pattern p = Pattern.compile(sensitiveWord);
		Matcher matcher = p.matcher(inputWords);
		inputWords = matcher.replaceAll("*");
		return inputWords;
	}

	/**
	 * @Title: getPattern
	 * @Description: 修改list为正则表达式
	 * @param list
	 * @return String
	 */
	public static String getPattern(List<String> list) {
		// 正则表达式串
		String pattern = "";
		if (null != list && list.size() > 0) {
			for (int i = 0; i < list.size(); i++) {
				String str = (String) list.get(i);
				if (str == null || "".equals(str)) {
					continue;
				}
				// 将字符加上正则表达式
				String temp = "";
				String[] strs = str.split(",");
				if (strs != null && strs.length > 0) {
					for (int k = 0; k < strs.length; k++) {
						String temp1 = "";
						if (strs[k] != null && !"".equals(strs[k])) {
							for (int j = 0; j < strs[k].length(); j++) {
								if (j == 0) {
									temp1 += strs[k].charAt(j);
									continue;
								}
								temp1 = temp1 + "\\s*" + strs[k].charAt(j);
							}
						}
						if (k == 0) {
							temp = temp1;
						} else {
							temp = temp + "|" + temp1;
						}
					}
				}
				if (i == 0) {
					pattern = temp;
				} else {
					pattern = pattern + "|" + temp;
				}
			}
		}
		return pattern;
	}

	/**
	 * @Title: main
	 * @Description: 使用方法
	 * @param args
	 */
	public static void main(String[] args) {
		SensitiveWordFilter.execFilter("垃 圾 fu 是ck sfshi t sf士fuk 大夫法轮功，不好");
	}
}
