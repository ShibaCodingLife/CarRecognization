package com.neusoft.core.util;

import static com.neusoft.core.util.Util.bytesToHex;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * @ClassName: MD5Util
 * @Description: MD5加密工具
 */
public class MD5Util {

	/**
	 * @Title: gen
	 * @Description: MD5加密
	 * @param datas
	 * @return String MD5码
	 */
	public static String gen(byte[] datas) {
		if (datas == null)
			return "md5 generate error!";
		return bytesToHex(encode(datas));
	}

	/**
	 * @Title: encode
	 * @Description: 加密执行方法
	 * @param datas
	 * @return byte[]
	 */
	private static byte[] encode(byte[] datas) {
		byte[] r = null;
		MessageDigest md;
		try {
			md = MessageDigest.getInstance("MD5");
			r = md.digest(datas);
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
		}
		return r;
	}
}
